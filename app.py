import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

import streamlit as st

from src.agent.graph import ask_agent


st.set_page_config(
    page_title="Travel Memory Agent",
    page_icon="🧳",
    layout="wide"
)


def get_existing_image_paths(evidence):
    
    image_paths = []
    seen_paths = set()

    for item in evidence:
        source = item.source or item.metadata.get("image_path", "")

        if not source:
            continue

        source = str(source).strip()

        if not source.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            continue

        image_path = PROJECT_ROOT / source

        if image_path.exists() and image_path not in seen_paths:
            image_paths.append(image_path)
            seen_paths.add(image_path)

    return image_paths


def display_related_images(evidence, max_images=2):
    
    image_paths = get_existing_image_paths(evidence)

    if not image_paths:
        return

    st.markdown("### Related image")

    for image_path in image_paths[:max_images]:
        st.image(
            str(image_path),
            caption=image_path.name,
            use_container_width=True
        )


st.title("🧳 Personalised Travel Memory Multimodal Agent")
st.caption("Text notes + image captions + metadata + LangGraph-style routing")

with st.sidebar:
    st.header("How to use")
    st.write(
        "Ask questions about your Auckland, Queenstown, Milford Sound, "
        "Mount Cook, Lake Tekapo, and Christchurch memories."
    )

    st.write("Example questions:")
    st.code("What did I eat at Sierra Café in Auckland?")
    st.code("Find the memory where the image shows a small Hobbit house.")
    st.code("Find an image which includes toast.")
    st.code("Compare Auckland and Queenstown. Which city gave me more landscape memories?")
    st.code("I want a scenic memory similar to Queenstown Skyline but less risky than Mount Victoria.")

    st.divider()

    st.write(
        "The system routes each question to text retrieval, image-caption retrieval, "
        "metadata retrieval, or hybrid retrieval."
    )


if "messages" not in st.session_state:
    st.session_state.messages = []

if "history" not in st.session_state:
    st.session_state.history = ""


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant" and "image_paths" in msg:
            for image_path in msg["image_paths"]:
                path_obj = Path(image_path)
                if path_obj.exists():
                    st.image(
                        str(path_obj),
                        caption=path_obj.name,
                        use_container_width=True
                    )


question = st.chat_input("Ask about your travel memories...")

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving travel memories..."):
            result = ask_agent(
                question,
                history=st.session_state.history
            )

        st.markdown(result["answer"])

        image_paths = get_existing_image_paths(result["evidence"])

        if image_paths:
            st.markdown("### Related image")
            for image_path in image_paths[:2]:
                st.image(
                    str(image_path),
                    caption=image_path.name,
                    use_container_width=True
                )

        with st.expander("Retrieval and grounding details"):
            st.write(f"**Route:** {result['route']}")
            st.write(result["verification"])

            for i, ev in enumerate(result["evidence"], start=1):
                st.markdown(f"### Evidence {i}: `{ev.doc_id}`")
                st.write(f"Modality: {ev.modality}")
                st.write(f"City: {ev.city}")
                st.write(f"Place: {ev.place}")
                st.write(f"Score: {ev.score:.3f}")
                st.write(f"Source/Image path: `{ev.source}`")
                st.text(ev.content[:1200])

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "image_paths": [str(path) for path in image_paths[:2]]
    })

    st.session_state.history += (
        f"\nUser: {question}\n"
        f"Assistant: {result['answer']}\n"
    )