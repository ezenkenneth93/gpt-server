from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.callbacks.base import BaseCallbackHandler
from dotenv import load_dotenv
import os
import json
import asyncio
from typing import Any, Dict, List

# 🔐 환경변수에서 OpenAI API 키 로드
load_dotenv()

# 🔧 FastAPI 앱 초기화
app = FastAPI()

# 🔧 CORS 설정 (개발 단계에서는 모든 도메인 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔧 스트리밍을 위한 커스텀 콜백 핸들러
class StreamingCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.tokens = []
        self.finished = False

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self.tokens.append(token)

    def on_llm_end(self, response, **kwargs: Any) -> None:
        self.finished = True

# 🔧 모델 설정
llm = ChatOpenAI(model="gpt-4", temperature=0.7)

# 🔧 스트리밍용 모델 설정
streaming_llm = ChatOpenAI(
    model="gpt-4", 
    temperature=0.7,
    streaming=True
)

# 🎯 프롬프트 템플릿 정의
full_prompt = PromptTemplate.from_template("""
당신은 영어를 잘하고 싶은 한국 부모들을 위한 친절하고 전문적인 영어 선생님입니다.
부모는 아이와 영어로 대화하기 위해 연습 중이며, 숙제처럼 영어 문장을 입력했습니다.
GPT의 역할은 다음과 같습니다:

1. 부모가 입력한 문장을 확인하고, 잘못된 부분이 있다면 자연스럽게 고쳐주세요. 
2. 수정한 이유를 한국 부모도 이해할 수 있게 쉽고 간단한 말로 설명해주세요.
3. 아이와의 일상 대화에서 함께 쓸 수 있는 관련 영어 표현 3가지를 제안해주세요. 
4. 만약 사용자가 질문을 함께 입력했다면, 그 질문에 대해 명확하게 답변해주세요.
   (질문은 선택 항목이므로, 입력이 없으면 생략하세요.)
5. 잘못된 부분을 고쳐주는것과 아이와 함께 쓸 수 있는 영어표현 3가지에 대해서 항상 그 뜻도 자세히 설명해주세요.
# 6. 가장 마지막에는 항상 잘 하고있다는, 혹은 다른 표현의 응원의 말로 마지막을 장식해주세요. 좀 부드럽게 응원해주세요.
아래는 사용자가 입력한 내용입니다:

문장: "{english_essay}"

질문 (선택): "{question}"

출력 형식은 다음과 같이 해주세요:

✅ 고쳐준 문장

🧠 수정한 이유

💬 함께 쓸 수 있는 표현 3가지

❓ 질문에 대한 답변 (질문이 있을 경우에만)
""")

feedback_chain = LLMChain(llm=llm, prompt=full_prompt)

# ✅ 입력 및 출력 데이터 모델 정의
class FeedbackRequest(BaseModel):
    english_essay: str
    question: str = ""

class FeedbackResponse(BaseModel):
    result: str

# ✅ 피드백 생성 API (기존)
@app.post("/generate-feedback", response_model=FeedbackResponse)
async def generate_feedback(data: FeedbackRequest):
    response = feedback_chain.invoke({
        "english_essay": data.english_essay,
        "question": data.question
    })
    return FeedbackResponse(result=response["text"])

# 🚀 스트리밍 피드백 생성 API (새로 추가)
@app.post("/stream-feedback")
async def stream_feedback(data: FeedbackRequest):
    async def generate():
        try:
            # 시작 메시지
            yield f"data: {json.dumps({'status': 'starting', 'message': 'AI 피드백 생성을 시작합니다...'}, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.1)
            
            # 스트리밍 콜백 핸들러 생성
            callback_handler = StreamingCallbackHandler()
            
            # 스트리밍용 LLM 체인 생성
            streaming_chain = LLMChain(
                llm=ChatOpenAI(
                    model="gpt-4", 
                    temperature=0.7,
                    streaming=True,
                    callbacks=[callback_handler]
                ), 
                prompt=full_prompt
            )
            
            # 진행 상태 메시지
            yield f"data: {json.dumps({'status': 'processing', 'message': 'GPT-4가 피드백을 생성하고 있습니다...'}, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.1)
            
            # 피드백 생성 시작
            response = await asyncio.to_thread(
                streaming_chain.invoke,
                {
                    "english_essay": data.english_essay,
                    "question": data.question
                }
            )
            
            # 전체 응답을 청크 단위로 전송
            full_text = response["text"]
            chunk_size = 100  # 한 번에 보낼 문자 수
            
            for i in range(0, len(full_text), chunk_size):
                chunk = full_text[i:i + chunk_size]
                chunk_data = json.dumps({"content": chunk}, ensure_ascii=False)
                yield f"data: {chunk_data}\n\n"
                await asyncio.sleep(0.05)  # 약간의 지연으로 스트리밍 효과
            
            # 완료 메시지
            yield f"data: {json.dumps({'done': True, 'message': '피드백 생성이 완료되었습니다.'}, ensure_ascii=False)}\n\n"
            
        except Exception as e:
            # 에러 처리
            error_data = json.dumps({"error": f"피드백 생성 중 오류가 발생했습니다: {str(e)}"}, ensure_ascii=False)
            yield f"data: {error_data}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type,Authorization",
        }
    )


# 테스트
async def stream_response():
    for i in range(5):
        yield f"data: {json.dumps({'content': f'Chunk {i}'}, ensure_ascii=False)}\n\n"
        await asyncio.sleep(1)

@app.post("/api/test-stream")
async def test_stream():
    return StreamingResponse(stream_response(), media_type="text/event-stream")