from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os

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

# 🔧 모델 설정
llm = ChatOpenAI(model="gpt-4", temperature=0.7)

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
6. 가장 마지막에는 항상 잘 하고있다는, 혹은 다른 표현의 응원의 말로 마지막을 장식해주세요. 좀 부드럽게 응원해주세요.
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

# ✅ 피드백 생성 API
@app.post("/generate-feedback", response_model=FeedbackResponse)
async def generate_feedback(data: FeedbackRequest):
    response = feedback_chain.invoke({
        "english_essay": data.english_essay,
        "question": data.question
    })
    return FeedbackResponse(result=response["text"])