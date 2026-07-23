import streamlit as st
import threading
from datetime import datetime
from dotenv import load_dotenv
from database.redis_cache import RedisCache
from memory.session_manager import SessionManager

load_dotenv()

st.set_page_config(page_title="智扫通 · 智能客服", page_icon="🤖")
st.title("🤖 智扫通机器人智能客服")
st.caption("基于 LangChain ReAct Agent + 四层记忆库")
st.divider()

redis_cache = RedisCache()
session_manager = SessionManager(redis_cache)

def get_agent():
    if "agent" not in st.session_state:
        from agent.react_agent import ReactAgent
        st.session_state["agent"] = ReactAgent()
    return st.session_state["agent"]

if "user_id" not in st.session_state:
    st.session_state["user_id"] = "user_default"

if "current_session_id" not in st.session_state:
    existing_sessions = session_manager.list_user_sessions(st.session_state["user_id"])
    if existing_sessions:
        latest_session = existing_sessions[0]
        st.session_state["current_session_id"] = latest_session["session_id"]
        loaded_data = session_manager.load_session(latest_session["session_id"])
        st.session_state["messages"] = loaded_data["messages"] if loaded_data else []
    else:
        st.session_state["current_session_id"] = session_manager.create_session(st.session_state["user_id"])
        st.session_state["messages"] = []

if "messages" not in st.session_state:
    st.session_state["messages"] = []

with st.sidebar:
    st.subheader("用户设置")
    user_id_input = st.text_input("用户ID", value=st.session_state["user_id"])
    if user_id_input != st.session_state["user_id"]:
        st.session_state["user_id"] = user_id_input
        existing_sessions = session_manager.list_user_sessions(user_id_input)
        if existing_sessions:
            latest_session = existing_sessions[0]
            st.session_state["current_session_id"] = latest_session["session_id"]
            loaded_data = session_manager.load_session(latest_session["session_id"])
            st.session_state["messages"] = loaded_data["messages"] if loaded_data else []
        else:
            st.session_state["current_session_id"] = session_manager.create_session(user_id_input)
            st.session_state["messages"] = []
        st.rerun()

    st.subheader("会话管理")
    if st.button("📝 新建会话"):
        st.session_state["current_session_id"] = session_manager.create_session(st.session_state["user_id"])
        st.session_state["messages"] = []
        st.rerun()

    st.subheader("历史会话")
    sessions = session_manager.list_user_sessions(st.session_state["user_id"])
    if sessions:
        for session in sessions:
            session_date = datetime.fromisoformat(session["saved_at"]).strftime("%m-%d %H:%M") if session["saved_at"] else ""
            if st.button(f"📄 {session['preview'][:20]}...\n{session_date}"):
                loaded_data = session_manager.load_session(session["session_id"])
                if loaded_data:
                    st.session_state["messages"] = loaded_data["messages"]
                    st.session_state["current_session_id"] = session["session_id"]
                    st.rerun()

for message in st.session_state["messages"]:
    if message["role"] == "assistant" and message.get("thoughts"):
        with st.chat_message("assistant"):
            with st.expander("🧠 查看思考过程", expanded=False):
                st.markdown(message["thoughts"], unsafe_allow_html=True)
            st.write(message["content"])
    else:
        st.chat_message(message["role"]).write(message["content"])

prompt = st.chat_input()

if prompt:
    st.chat_message("user").write(prompt)
    st.session_state["messages"].append({"role": "user", "content": prompt})

    thoughts_html = ""
    response_container = st.chat_message("assistant")
    response_placeholder = response_container.empty()
    
    with st.expander("🧠 查看思考过程", expanded=True):
        thoughts_placeholder = st.empty()
    
    full_response = ""
    
    res_stream = get_agent().execute_stream(
        prompt,
        user_id=st.session_state["user_id"],
        session_id=st.session_state["current_session_id"]
    )

    for chunk in res_stream:
        if chunk.startswith("\n\U0001f91d [派发给:"):
            thoughts_html += f"<div style='color: #9C27B0; margin: 5px 0;'>🤝 {chunk.replace(chr(10), '<br>')}</div>"
        elif chunk.startswith("\n\U0001f527 [调用工具"):
            thoughts_html += f"<div style='color: #FF9800; margin: 5px 0;'>🔧 {chunk.replace(chr(10), '<br>')}</div>"
        elif chunk.startswith("\n\u2705 [工具返回"):
            thoughts_html += f"<div style='color: #2196F3; margin: 5px 0;'>✅ {chunk.replace(chr(10), '<br>')}</div>"
        elif chunk.startswith("\n\u2705 [") and "专家返回" in chunk:
            thoughts_html += f"<div style='color: #009688; margin: 5px 0;'>✅ {chunk.replace(chr(10), '<br>')}</div>"
        else:
            full_response += chunk

        if thoughts_html:
            thoughts_placeholder.markdown(thoughts_html, unsafe_allow_html=True)

        if full_response:
            response_placeholder.write(full_response)

    if full_response:
        st.session_state["messages"].append({
            "role": "assistant",
            "content": full_response,
            "thoughts": thoughts_html
        })
    
    session_manager.save_session(st.session_state["current_session_id"], st.session_state["messages"], user_id=st.session_state["user_id"])
    
    st.rerun()