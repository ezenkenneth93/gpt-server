from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.callbacks.base import BaseCallbackHandler
from dotenv import load_dotenv
from fastapi import WebSocket, WebSocketDisconnect
import os
import json
import asyncio

from asyncio import Queue
from typing import Any

# 🔐 환경변수에서 OpenAI API 키 로드
load_dotenv()

# 🔧 FastAPI 앱 초기화
app = FastAPI()

# 🔧 CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔧 스트리밍을 위한 콜백 핸들러 (비동기 큐 방식)
class AsyncQueueCallbackHandler(BaseCallbackHandler):
    def __init__(self, queue: Queue):
        self.queue = queue
        self.loop = asyncio.get_event_loop()

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        asyncio.run_coroutine_threadsafe(self.queue.put(token), self.loop)

    def on_llm_end(self, response, **kwargs: Any) -> None:
        asyncio.run_coroutine_threadsafe(self.queue.put(None), self.loop)

# 🔧 일반 LLM (기본)
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.7
)

# 🔧 전역 스트리밍 LLM 인스턴스 (콜백 없이 재사용)
base_streaming_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    streaming=True
)

# ✅ 콜백만 다르게 설정하여 새로운 인스턴스를 만드는 함수
def get_llm_with_callback(callback_handler: BaseCallbackHandler) -> ChatOpenAI:
    return ChatOpenAI(
        model=base_streaming_llm.model_name,
        temperature=base_streaming_llm.temperature,
        streaming=True,
        callbacks=[callback_handler]
    )

# 🔧 프롬프트 템플릿 정의
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
6. 가장 마지막에는 항상 잘 하고있다는, 혹은 다른 표현의 응원의 말로 마지막을 장식해주세요. 좀 부드럽게 응원해주세요.
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

# ✅ 일반 피드백용 체인
feedback_chain = LLMChain(llm=llm, prompt=full_prompt)

# ✅ 데이터 모델 정의
class FeedbackRequest(BaseModel):
    english_essay: str
    question: str = ""

class FeedbackResponse(BaseModel):
    result: str

# ✅ 일반 피드백 API
@app.post("/generate-feedback", response_model=FeedbackResponse)
async def generate_feedback(data: FeedbackRequest):
    response = feedback_chain.invoke({
        "english_essay": data.english_essay,
        "question": data.question
    })
    return FeedbackResponse(result=response["text"])

# ✅ 스트리밍 피드백 API
# ✅ 스트리밍 피드백 API
@app.post("/stream-feedback")
async def stream_feedback(data: FeedbackRequest):
    print("🔥 요청 수신됨!")
    print("📝 Essay:", data.english_essay[:50])  # 일부만 출력
    print("❓ Question:", data.question[:50])    # 일부만 출력

    async def generate():
        try:
            print("⚙️ 스트리밍 생성기 시작됨")

            yield f"data: {json.dumps({'status': 'connected', 'message': '연결 완료! 피드백 생성 준비 중...'}, ensure_ascii=False)}\n\n"

            queue_instance = Queue()
            callback_handler = AsyncQueueCallbackHandler(queue_instance)

            streaming_llm_with_callback = get_llm_with_callback(callback_handler)
            print("✅ LLM 콜백 핸들러 세팅 완료")

            yield f"data: {json.dumps({'status': 'ready', 'message': '설정 완료! AI 분석 시작...'}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'status': 'processing', 'message': 'GPT가 피드백을 생성 중입니다...'}, ensure_ascii=False)}\n\n"

            prompt_text = full_prompt.format(
                english_essay=data.english_essay,
                question=data.question
            )
            print("📤 프롬프트 생성 완료")

            # 체인 실행 (쓰레드에서)
            async def run_chain():
                print("🚀 GPT 호출 시작")
                await asyncio.to_thread(
                    streaming_llm_with_callback.invoke,
                    prompt_text
                )
                print("✅ GPT 호출 완료")

            asyncio.create_task(run_chain())

            # 큐에서 토큰 수신
            while True:
                token = await queue_instance.get()
                # print("📥 수신된 토큰:", token)
                if token is None:
                    print("🔚 토큰 수신 종료")
                    break
                yield f"data: {json.dumps({'content': token}, ensure_ascii=False)}\n\n"

            yield f"data: {json.dumps({'done': True, 'message': '피드백 생성이 완료되었습니다.'}, ensure_ascii=False)}\n\n"
            print("✅ 스트리밍 완료")

        except Exception as e:
            print("❌ 예외 발생:", str(e))
            yield f"data: {json.dumps({'error': f'피드백 생성 중 오류: {str(e)}'}, ensure_ascii=False)}\n\n"

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


# ✅ 예열
@app.on_event("startup")
async def startup_event():
    try:
        print("🔥 Pre-warming: OpenAI 연결 준비 중...")
        llm.invoke("Hello")
        print("✅ Pre-warming 완료: 모델 연결 성공!")
    except Exception as e:
        print(f"⚠️ Pre-warming 실패: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    print("🔄 앱 종료 중...")
    print("✅ 정리 완료!")

# websocket 연결
@app.websocket("/ws-feedback")
async def websocket_feedback(websocket: WebSocket):
    await websocket.accept()

    try:
        # 클라이언트가 보낸 JSON 파싱
        data = await websocket.receive_text()
        payload = json.loads(data)
        essay = payload.get("english_essay", "")
        question = payload.get("question", "")

        # 큐/콜백 구성
        queue_instance = Queue()
        callback_handler = AsyncQueueCallbackHandler(queue_instance)
        streaming_llm_with_callback = get_llm_with_callback(callback_handler)

        # 프롬프트 렌더링
        prompt_text = full_prompt.format(
            english_essay=essay,
            question=question
        )

        # LLM 실행 (비동기)
        async def run_chain():
            await asyncio.to_thread(
                streaming_llm_with_callback.invoke,
                prompt_text
            )

        asyncio.create_task(run_chain())

        # 큐에서 토큰 수신 → WebSocket 전송
        while True:
            token = await queue_instance.get()
            if token is None:
                break
            await websocket.send_text(token)

        await websocket.send_text("__END__")

    except WebSocketDisconnect:
        print("❌ 클라이언트 연결 해제됨")
    except Exception as e:
        await websocket.send_text(f"__ERROR__: {str(e)}")
        await websocket.close()
