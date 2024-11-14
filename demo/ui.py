import json
import uuid
from datetime import datetime

import streamlit as st
from pydantic import BaseModel

from peony.adapters import postgres as pg
from peony.client import Peony


class PreferenceModel(BaseModel):
    first_response: str
    second_response: str | None


MODEL_MAPPING = {
    "gpt-4o": "openai/gpt-4o-mini",
    "claude-3.5": "anthropic/claude-3-5-sonnet-20240620",
}

client = Peony()
pClient = pg.PostgresAdapter(
    host="postgres_dev",
    dbname="postgres",
    user="postgres",
    pw="postgres",
    port="5432",
)


PROMPT = """
Answer user questions with clear answer.
Make your answer short but good quality.
"""


def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = {}
    if "context" not in st.session_state:
        st.session_state.context = [{"role": "system", "content": PROMPT}]
    if "cur_vote" not in st.session_state:
        st.session_state.cur_vote = ""
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
        "alt_content": alt_content if alt_content else None,
        "timestamp": datetime.now().strftime("%H:%M"),
        "has_alt": alt_content is not None,
    }

    # preserve context
    st.session_state.context.append({"role": role, "content": content})

    if alt_content:
        # need record in the state for first pass
        if message_id not in st.session_state.votes:
            st.session_state.votes[message_id] = {
                "voted": False,
                "preferred": None,
            }

    st.session_state.counter += 1


def process_messages(messages, model, diff_frequency):
    last_message_id = max(messages.keys())
    last_message = messages[last_message_id]

    if last_message["role"] == "user":
        print("sending request to model")

        with st.spinner(text="model processing message..."):
            response = client.chat.completions.create_preference(
                model=MODEL_MAPPING[model],
                messages=[
                    *st.session_state.context,
                    {"role": "user", "content": last_message["content"]},
                ],
                response_format=PreferenceModel,
                diff_frequency=diff_frequency,
            )

            res = json.loads(response.choices[0].message.content)
            first_res = res["first_response"]
            second_res = res["second_response"]

        add_message("assistant", first_res, alt_content=second_res)


def handle_vote(message_id, preferred):
    if not st.session_state.votes[message_id]["voted"]:
        st.session_state.votes[message_id]["voted"] = True
        st.session_state.votes[message_id]["preferred"] = preferred


def handle_input(model, diff_frequency):
    if st.session_state.user_input.strip():
        message = st.session_state.user_input
        add_message("user", message)

    messages = st.session_state.messages
    if not messages:
        return

    process_messages(messages, model, diff_frequency)


def save_preference(message_id, preferred, rejected):
    prev_msg = st.session_state.messages[message_id - 1]
    prompt = prev_msg["content"] if prev_msg["role"] == "user" else ""

    data = {
        "id": uuid.uuid1(),
        "prompt": prompt,
        "preferred": preferred,
        "rejected": rejected,
    }

    st.session_state.pref_data.append(data)


def create_sidebar():
    st.sidebar.header("Settings")

    model = st.sidebar.selectbox(
        label="Model",
        options=["gpt-4o", "claude-3.5"],
        index=0,
    )

    diff_frequency = st.sidebar.slider(
        label="Difference Frequency",
        min_value=0.0,
        max_value=1.0,
        value=0.4,
        help="How often model to show comparison responses.",
    )

    return model, diff_frequency


def main():
    st.set_page_config(page_title="Peony Chat")

    model, diff_frequency = create_sidebar()

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


        /* Add padding at bottom */
        .main-container {
            padding-bottom: 5rem;
        }

        /* Style vote buttons */
        .stButton button {
            width: 100%;
            margin-top: 0.5rem;
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

            # display message history
            message_items = list(st.session_state.messages.items())

            if message_items:
                for message_id, message in message_items[: len(message_items) - 1]:
                    if message["role"] == "user":
                        with st.chat_message("user"):
                            st.markdown(message["content"])

                    else:
                        # Assistant message(s)
                        if message["has_alt"]:
                            vote_state = st.session_state.votes[message_id]
                            selected_content = (
                                message["content"]
                                if vote_state["preferred"] == "A"
                                else message["alt_content"]
                            )

                            with st.chat_message("assistant"):
                                st.markdown(selected_content)

                        else:
                            # Single response
                            with st.chat_message("assistant"):
                                st.markdown(message["content"])

                # stream last message
                last_msg_id, last_msg = message_items[-1]

                if last_msg["role"] == "user":
                    with st.chat_message("user"):
                        st.markdown(last_msg["content"])

                else:
                    if last_msg["has_alt"]:
                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("### Response A")

                            with st.chat_message("assistant"):
                                st.markdown(last_msg["content"])

                            if st.button(
                                "Vote for A",
                                key=f"vote_a_{last_msg_id}",
                            ):
                                st.session_state.cur_vote = "A"
                                handle_vote(last_msg_id, "A")
                                save_preference(
                                    last_msg_id,
                                    last_msg["content"],
                                    last_msg["alt_content"],
                                )

                                pClient.save_feedback(
                                    table="preference",
                                    data=st.session_state.pref_data,
                                )

                        with col2:
                            st.markdown("### Response B")

                            with st.chat_message("assistant"):
                                st.markdown(last_msg["alt_content"])

                            if st.button(
                                "Vote for B",
                                key=f"vote_b_{last_msg_id}",
                            ):
                                st.session_state.cur_vote = "B"
                                handle_vote(last_msg_id, "B")
                                save_preference(
                                    last_msg_id,
                                    last_msg["alt_content"],
                                    last_msg["content"],
                                )

                                pClient.save_feedback(
                                    table="preference",
                                    data=st.session_state.pref_data,
                                )

                    else:
                        with st.chat_message("assistant"):
                            st.markdown(last_msg["content"])

            _ = st.chat_input(
                placeholder="Type your message here...",
                key="user_input",
                on_submit=handle_input,
                args=(model, diff_frequency),
            )


if __name__ == "__main__":
    main()
