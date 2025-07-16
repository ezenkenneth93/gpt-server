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

from asyncio import create_task
from asyncio import Queue   
from typing import Any, Dict, List

# 원래 상태로 복원

# 🔐 환경변수에서 OpenAI API 키 로드
load_dotenv()

# 🔧 OPTION 4: 연결 풀링은 일단 제거 (호환성 문제)

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

# 🔧 스트리밍을 위한 커스텀 콜백 핸들러 (백업용 - 기존 가짜 스트리밍)
# class StreamingCallbackHandler(BaseCallbackHandler):
#     def __init__(self):
#         self.tokens = []
#         self.finished = False

#     def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
#         self.tokens.append(token)

#     def on_llm_end(self, response, **kwargs: Any) -> None:
#         self.finished = True

# 🚀 진짜 실시간 스트리밍을 위한 비동기 큐 콜백 핸들러
class AsyncQueueCallbackHandler(BaseCallbackHandler):
    def __init__(self, queue: Queue):
        self.queue = queue
        self.loop = asyncio.get_event_loop()  # 메인 루프 저장

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        # 메인 루프에 코루틴 안전하게 스레드에서 실행
        asyncio.run_coroutine_threadsafe(self.queue.put(token), self.loop)

    def on_llm_end(self, response, **kwargs: Any) -> None:
        asyncio.run_coroutine_threadsafe(self.queue.put(None), self.loop)


# 🔧 모델 설정
llm = ChatOpenAI(
    model="gpt-4o", 
    temperature=0.7
)

# 🔧 스트리밍용 모델 설정 (전역 인스턴스 - 재사용)
streaming_llm = ChatOpenAI(
    model="gpt-4o", 
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
7. 응답을 출력하는 에디터는 마크업, 마크다운이 아니므로 강조할 떄 ** 를 사용하지 말아주세요.
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

# 🚀 OPTION 9: Pre-warming (예열) - 앱 시작 시 연결 미리 준비
@app.on_event("startup")
async def startup_event():
    try:
        print("🔥 Pre-warming: OpenAI 연결 준비 중...")
        # 더미 요청으로 연결 예열
        dummy_response = llm.invoke("Hello")
        print("✅ Pre-warming 완료: OpenAI 연결 준비됨!")
    except Exception as e:
        print(f"⚠️ Pre-warming 실패 (정상 동작에는 영향 없음): {e}")

@app.on_event("shutdown")
async def shutdown_event():
    print("🔄 앱 종료 중...")
    print("✅ 정리 완료!")

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
            # 🚀 OPTION 10: 즉시 응답 시작 - 연결 확인 메시지부터 전송
            yield f"data: {json.dumps({'status': 'connected', 'message': '연결 완료! 피드백 생성 준비 중...'}, ensure_ascii=False)}\n\n"
            
            # 스트리밍 콜백 핸들러 생성
            # callback_handler = StreamingCallbackHandler() # 기존 가짜 스트리밍 핸들러
            queue_instance = Queue()   # 진짜 실시간 스트리밍 큐 (thread-safe)
            callback_handler = AsyncQueueCallbackHandler(queue_instance)
            
            # 🚀 콜백이 포함된 스트리밍 LLM 생성 (요청마다 새로 생성, 콜백 포함)
            streaming_llm_with_callback = ChatOpenAI(
                model="gpt-4o", 
                temperature=0.7,
                streaming=True,
                callbacks=[callback_handler]  # ChatOpenAI 레벨에서 콜백 설정
            )
            
            # 설정 완료 메시지
            yield f"data: {json.dumps({'status': 'ready', 'message': '설정 완료! AI 분석 시작...'}, ensure_ascii=False)}\n\n"
            
            # 🚀 스트리밍용 LLM 체인 생성
            streaming_chain = LLMChain(
                llm=streaming_llm_with_callback,  # 콜백이 포함된 인스턴스 사용
                prompt=full_prompt
            )
            
            # 진행 상태 메시지
            yield f"data: {json.dumps({'status': 'processing', 'message': 'GPT-4 Turbo가 피드백을 생성하고 있습니다...'}, ensure_ascii=False)}\n\n"
            
            # 🚀 진짜 실시간 스트리밍: LLM 체인을 별도 태스크에서 실행
            async def run_chain():
                await asyncio.to_thread(
                    streaming_chain.invoke,
                    {
                        "english_essay": data.english_essay,
                        "question": data.question
                    }
                )
            
            # 체인 실행 시작
            chain_task = asyncio.create_task(run_chain())
            
            # 🚀 토큰 스트리밍: 큐에서 토큰을 받는 즉시 바로 전송
            while True:
                # thread-safe queue에서 non-blocking으로 가져오기
                token = await queue_instance.get()
                if token is None:  # 종료 신호
                    break
                yield f"data: {json.dumps({'content': token}, ensure_ascii=False)}\n\n"
            
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