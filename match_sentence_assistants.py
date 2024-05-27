from typing import List
import streamlit as st
import os
# from langchain.agents.openai_assistant import OpenAIAssistantRunnable
# from langchain.prompts import (
#     ChatPromptTemplate,
#     HumanMessagePromptTemplate,
#     PromptTemplate,
# )
# from langchain.schema import StrOutputParser
# from langchain_core.output_parsers import JsonOutputParser
# import langchain_core.pydantic_v1 as pyd1

from typing_extensions import override
from openai import AssistantEventHandler

# First, we create a EventHandler class to define
# how we want to handle the events in the response stream.


class EventHandler(AssistantEventHandler):

    result = ""

    @override
    def on_text_created(self, text) -> None:
        print(f"\nassistant > ", end="", flush=True)

    @override
    def on_text_delta(self, delta, snapshot):
        print(delta.value, end="", flush=True)
        EventHandler.result += delta.value
        message_placeholder.markdown(EventHandler.result + "▌")
        

    def on_tool_call_created(self, tool_call):
        print(f"\nassistant > {tool_call.type}\n", flush=True)
        EventHandler.result += f"{tool_call.type}\n\n```"
        message_placeholder.markdown(EventHandler.result + "▌")

    def on_tool_call_delta(self, delta, snapshot):
        if delta.type == "code_interpreter":
            if delta.code_interpreter.input:
                print(delta.code_interpreter.input, end="", flush=True)
                EventHandler.result += delta.code_interpreter.input
                message_placeholder.markdown(EventHandler.result + "▌")
            if delta.code_interpreter.outputs:
                print(f"\noutput >", flush=True)
                EventHandler.result += f"\n```\n\n\noutput >\n\n"
                message_placeholder.markdown(EventHandler.result + "▌")
                for output in delta.code_interpreter.outputs:
                    if output.type == "logs":
                        print(f"\n{output.logs}", flush=True)
                        EventHandler.result += f"\n{output.logs}\n\n"
                        message_placeholder.markdown(EventHandler.result + "▌")

from openai import OpenAI
import time

def run_and_wait(client, assistant, thread):
  run = client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=assistant.id
  )
  while True:
    run_check = client.beta.threads.runs.retrieve(
      thread_id=thread.id,
      run_id=run.id
    )
    print(run_check.status)
    if run_check.status in ['queued','in_progress']:
      time.sleep(2)
    else:
      break
  return run

client = OpenAI(api_key=st.secrets.OPENAI_API_KEY)


# Streamlit 페이지 설정
st.set_page_config(page_title="AI English Sentence Compare Assistant", layout="wide")


# class Match(pyd1.BaseModel):
#     point: str = pyd1.Field(
#         description="기준 문장과 비교 문장을 비교하여 일치성을 백분율로 알려 준다."
#     )


# def build_match_chain(assistant):
#     parser = JsonOutputParser(pydantic_object=Match)
#     format_instruction = parser.get_format_instructions()

#     human_msg_prompt_template = HumanMessagePromptTemplate(
#         prompt=PromptTemplate(
#             template="기준 문장: '{reference}'\n비교 문장: '{input}'\n",  # ---\n다음의 포맷에 맞춰 응답해라. : {format_instruction}",
#             input_variables=["input", "reference"],
#             # partial_variables={"format_instruction": format_instruction},
#         )
#     )

#     prompt_template = ChatPromptTemplate.from_messages(
#         [
#             human_msg_prompt_template,
#         ],
#     )

#     chain = prompt_template | assistant | StrOutputParser()  # | parser
#     return chain


if "assistant" not in st.session_state:
    #     assistant = OpenAIAssistantRunnable.create_assistant(
    #         name="sentence compare assistants",
    #         instructions="""기준 영어 문장과 비교 영어 문장을 단어 단위로 비교하는 공식을 적용하여
    # 일치성을 기준 문장의 단어 수를 기준으로 백분율로 알려 준다.
    # 단, 비교 결과에 일관성이 있어야 한다.
    # 다음 문장이 이전에 했던 것과 같은 경우에는 같은 값으로 응답한다.""",
    #         tools=[],
    #         model="gpt-4o",
    #         file_ids=None,make
    #     )
    assistant = client.beta.assistants.retrieve(
        st.secrets.assistant_id
    )
    st.session_state.assistant = assistant

# if "match_chain" not in st.session_state:
#     st.session_state.match_chain = build_match_chain(st.session_state.assistant)


# 메인 섹션
st.title("AI 문장 비교 서비스")


reference_input = st.text_area("Enter the reference text here:")

# 사용자 입력을 위한 텍스트 에어리어
# user_input = st.text_area("Enter your text here:", value="Yesterday, I goes to the store for bought some milk.")
user_input = st.text_area("Enter your text here:")

st.button("분석하기")

if user_input:

    st.subheader("문장 일치")
    with st.container(border=True):
        with st.spinner("문장 일치 분석중..."):
            # match = st.session_state.match_chain.invoke(
            #    {"reference": reference_input, "input": user_input}
            # )
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                
            thread = client.beta.threads.create(
                messages=[
                    {
                        "role": "user",
                        "content": f"기준 문장: '{reference_input}'\n비교 문장: '{user_input}'\n",
                    }
                ]
            )
            with client.beta.threads.runs.stream(
                thread_id=thread.id,
                assistant_id=st.session_state.assistant.id,
                event_handler=EventHandler(),
            ) as stream:
                stream.until_done()

            # run = run_and_wait(client=client, assistant=st.session_state.assistant, thread=thread)
            # match = st.session_state.assistatnt.invoke(
            #     {
            #         "content": f"기준 문장: '{reference_input}'\n비교 문장: '{user_input}'\n"
            #     }
            # )
        
            # thread_messages = client.beta.threads.messages.list(thread.id)
            # for msg in reversed(thread_messages.data):
            #    print(f"{msg.role}({msg.created_at}): {msg.content[0].text.value}")
            #    if msg.role == "assistant":
            #       match += msg.content[0].text.value 

            # st.markdown(EventHandler.result)  # ["point"])
                message_placeholder.markdown(EventHandler.result)
        
