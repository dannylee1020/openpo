import json
import random
import uuid
from datetime import datetime

import streamlit as st
from client import CustomClient
from internal import storage
from pydantic import BaseModel

client = CustomClient()

PROMPT = "You are a helpful assistant. Help your users with concise and clear answers."


class ResponseModel(BaseModel):
    first_response: str
    second_response: str


def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = {}
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""
    if "votes" not in st.session_state:
        st.session_state.votes = {}
    if "pref_data" not in st.session_state:
        st.session_state.pref_data = []
    if "counter" not in st.session_state:
        st.session_state.counter = 0


def add_message(role, content, alt_content=None):
    message_id = st.session_state.counter

    st.session_state.messages[message_id] = {
        "role": role,
        "content": content,
        "alt_content": alt_content,
        "timestamp": datetime.now().strftime("%H:%M"),
        "has_alt": alt_content is not None,
    }

    if alt_content:
        # need record in the state for first pass
        if message_id not in st.session_state.votes:
            st.session_state.votes[message_id] = {
                "voted": False,
                "preferred": None,
            }

    st.session_state.counter += 1


def handle_vote(message_id, preferred):
    if not st.session_state.votes[message_id]["voted"]:
        st.session_state.votes[message_id]["voted"] = True
        st.session_state.votes[message_id]["preferred"] = preferred


def save_pref(message_id, preferred, rejected):
    prev_msg = st.session_state.messages[message_id - 1]
    prompt = prev_msg["content"] if prev_msg["role"] == "user" else ""

    data = {
        "id": uuid.uuid1(),
        "prompt": prompt,
        "preferred": preferred,
        "rejected": rejected,
    }

    st.session_state.pref_data.append(data)


def handle_input():
    if st.session_state.user_input.strip():
        message = st.session_state.user_input
        add_message("user", message)


def process_message():
    messages = st.session_state.messages
    if not messages:
        return

    last_message_id = max(messages.keys())
    last_message = messages[last_message_id]

    if last_message["role"] == "user":
        print("sending request to openai")

        response = client.beta.chat.completions.parse_preference(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": last_message["content"]},
            ],
            diff_frequency=0.2,
        )

        print("done getting response from openai")

        res = json.loads(response.choices[0].message.content)
        first_res = res["first_response"]
        second_res = res["second_response"] if len(res.keys()) > 1 else None

        add_message("assistant", first_res, alt_content=second_res)
        # Clear input
        st.session_state.user_input = ""


def main():
    st.set_page_config(page_title="Peony Chat")

    # Custom CSS combining both styles
    st.markdown(
        """
        <style>
        /* Container styles */
        .main-content {
            max-width: 800px;
            margin: 0 auto;
            padding: 1rem;
        }

        .chat-container {
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
        }

        .user-message {
            background-color: #e9ecef;
            margin-left: 20%;
        }

        .assistant-message {
            background-color: #f8f9fa;
            margin-right: 20%;
        }

        /* Comparison styles */
        .alternative-responses {
            display: flex;
            gap: 1rem;
            margin-bottom: 1rem;
        }

        .response-option {
            flex: 1;
            padding: 1rem;
            border-radius: 10px;
            background-color: #f8f9fa;
            border: 2px solid #dee2e6;
        }

        .response-option.selected {
            border-color: #28a745;
        }

        .vote-results {
            margin-top: 0.5rem;
            font-size: 0.9rem;
            color: #6c757d;
            text-align: center;
        }

        .timestamp {
            color: #6c757d;
            font-size: 0.8rem;
            margin-top: 0.2rem;
        }

        /* Input container styles */
        .input-container {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: white;
            padding: 1rem 0;
            border-top: 1px solid #e9ecef;
        }

        .input-container > div {
            max-width: 800px;
            margin: 0 auto;
            padding: 0 1rem;
        }

        .stTextInput input {
            max-width: 800px;
            border-radius: 10px;
            border: 1px solid #e9ecef;
            padding: 0.5rem 1rem;
        }

        /* Add padding at bottom */
        .main-container {
            padding-bottom: 5rem;
        }

        /* Make comparison columns more contained */
        .comparison-container {
            max-width: 800px;
            margin: 0 auto;
        }

        /* Style vote buttons */
        .stButton button {
            width: 100%;
            margin-top: 0.5rem;
        }

        @media (max-width: 768px) {
            .main-content {
                padding: 0.5rem;
            }
            .stTextInput input {
                max-width: 95%;
            }
        }
        </style>
    """,
        unsafe_allow_html=True,
    )

    # Initialize session state
    init_session_state()

    # Main container
    with st.container():
        st.markdown('<div class="main-content">', unsafe_allow_html=True)
        st.title("Peony Chat 🌸")
        # Messages container
        with st.container():
            st.markdown('<div class="main-container">', unsafe_allow_html=True)
            # display message

            with st.spinner("waiting for the response..."):
                process_message()

            for message_id, message in st.session_state.messages.items():
                if message["role"] == "user":
                    # User message
                    st.markdown(
                        f"""
                        <div class="chat-container user-message">
                            <div>{message["content"]}</div>
                            <div class="timestamp">{message["timestamp"]}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    # Assistant message(s)
                    if message["has_alt"]:
                        st.markdown(
                            '<div class="comparison-container">', unsafe_allow_html=True
                        )
                        col1, col2 = st.columns(2)

                        vote_state = st.session_state.votes[message_id]

                        with col1:
                            st.markdown("### Response A")
                            st.markdown(
                                f"""
                                <div class="response-option {'selected' if vote_state['preferred'] == 'A' else ''}">
                                    {message["content"]}
                                    <div class="timestamp">{message["timestamp"]}</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                            if not vote_state["voted"]:
                                if st.button("Vote for A", key=f"vote_a_{message_id}"):
                                    handle_vote(message_id, "A")

                                    save_pref(
                                        message_id,
                                        message["content"],
                                        message["alt_content"],
                                    )

                                    with storage.get_db_conn() as conn:
                                        storage.save_to_db(
                                            conn, st.session_state.pref_data
                                        )
                        with col2:
                            st.markdown("### Response B")
                            st.markdown(
                                f"""
                                <div class="response-option {'selected' if vote_state['preferred'] == 'B' else ''}">
                                    {message["alt_content"]}
                                    <div class="timestamp">{message["timestamp"]}</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                            if not vote_state["voted"]:
                                if st.button("Vote for B", key=f"vote_b_{message_id}"):
                                    handle_vote(message_id, "B")

                                    save_pref(
                                        message_id,
                                        message["alt_content"],
                                        message["content"],
                                    )

                                    with storage.get_db_conn() as conn:
                                        storage.save_to_db(
                                            conn, st.session_state.pref_data
                                        )
                        st.markdown("</div>", unsafe_allow_html=True)
                    else:
                        # Single response
                        st.markdown(
                            f"""
                            <div class="chat-container assistant-message">
                                <div>{message["content"]}</div>
                                <div class="timestamp">{message["timestamp"]}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Input container
    st.markdown(
        """
        <div class="input-container">
            <div>
    """,
        unsafe_allow_html=True,
    )

    user_query = st.text_input(
        "Message",
        key="user_input",
        on_change=handle_input,
        placeholder="Type your message here...",
    )

    st.markdown(
        """
            </div>
        </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
