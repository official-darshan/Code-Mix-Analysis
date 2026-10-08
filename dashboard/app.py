from __future__ import annotations

import os

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API_URL = os.getenv("SUPPORT_API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Support Conversation Analytics",
    page_icon="💬",
    layout="wide",
)

st.title("💬 Code-Mixed Support Conversation Analytics")
st.caption("Intent • Sentiment • Resolution • Escalation • Priority")

st.sidebar.header("System")
st.sidebar.write("Streamlit → FastAPI → ML → SQLite")
st.sidebar.write(f"API: {API_URL}")


def call_api(message: str, agent_response: str):
    response = requests.post(
        f"{API_URL}/analyze",
        json={
            "message": message,
            "agent_response": agent_response,
        },
        timeout=30,
    )

    if response.status_code != 200:
        try:
            detail = response.json().get("detail", "Unknown API error")
        except Exception:
            detail = response.text or "Unknown API error"
        raise RuntimeError(detail)

    return response.json()


def get_analytics():
    response = requests.get(f"{API_URL}/analytics", timeout=15)
    response.raise_for_status()
    return response.json().get("records", [])


tab1, tab2 = st.tabs(["🔎 Analyze Conversation", "📊 Analytics Dashboard"])

with tab1:
    st.subheader("Enter a customer conversation")

    message = st.text_area(
        "Customer message",
        value="Mera payment deduct ho gaya but order confirm nahi hua",
        height=120,
    )

    agent_response = st.text_area(
        "Agent response (optional, recommended for resolution)",
        value="",
        height=100,
        help=(
            "Resolution prediction is stronger when the agent response is also provided."
        ),
    )

    if st.button("Analyze Conversation", type="primary"):
        if len(message.strip()) < 3:
            st.error("Please enter at least 3 meaningful characters.")
        else:
            try:
                result = call_api(
                    message.strip(),
                    agent_response.strip(),
                )

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Language", result["language"])
                c2.metric("Intent", result["intent"])
                c3.metric("Sentiment", result["sentiment"])
                c4.metric("Priority", result["priority"])

                st.divider()

                c5, c6, c7, c8 = st.columns(4)
                c5.metric("Resolution", result["resolution"])
                c6.metric("Escalation", result["escalation"])
                c7.metric(
                    "Escalation Probability",
                    f"{result['escalation_probability'] * 100:.1f}%",
                )
                c8.metric(
                    "Confidence",
                    f"{result['confidence'] * 100:.1f}%",
                )

                if not agent_response.strip():
                    st.info(
                        "No agent response was provided. Resolution uses customer-message cues only."
                    )

                if result["status"] != "OK":
                    st.warning(result["status"])
                else:
                    st.success("Analysis completed and saved to SQLite database.")

                with st.expander("Model confidence and decision sources"):
                    st.write("Intent source:", result["intent_source"])
                    st.write("Sentiment source:", result["sentiment_source"])
                    st.write("Resolution source:", result["resolution_source"])
                    st.write("Escalation source:", result["escalation_source"])
                    st.json(
                        {
                            "intent_confidence": result["intent_confidence"],
                            "sentiment_confidence": result["sentiment_confidence"],
                            "resolution_confidence": result["resolution_confidence"],
                            "escalation_confidence": result["escalation_confidence"],
                        }
                    )

            except requests.exceptions.ConnectionError:
                st.error(
                    "FastAPI is not running. Start it with: "
                    "uvicorn api.main:app --reload"
                )
            except requests.exceptions.Timeout:
                st.error("The API took too long to respond. Please try again.")
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")

with tab2:
    st.subheader("Analytics Dashboard")

    try:
        records = get_analytics()
    except requests.exceptions.ConnectionError:
        st.error("FastAPI is not running. Start it first.")
        records = []
    except Exception as exc:
        st.error(f"Could not load analytics: {exc}")
        records = []

    if not records:
        st.info("No analyzed conversations yet. Analyze a message first.")
    else:
        df = pd.DataFrame(records)

        total = len(df)
        resolved = int((df["resolution"] == "Resolved").sum())
        escalated = int((df["escalation"] == "Yes").sum())
        high = int((df["priority"] == "High").sum())
        review = int((df["status"] == "Manual Review Recommended").sum())

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total", total)
        c2.metric("Resolved", resolved)
        c3.metric("Escalated", escalated)
        c4.metric("High Priority", high)
        c5.metric("Needs Review", review)

        col1, col2 = st.columns(2)

        with col1:
            counts = df["intent"].value_counts().reset_index()
            counts.columns = ["intent", "count"]
            fig = px.bar(counts, x="intent", y="count", title="Intent Distribution")
            fig.update_layout(xaxis_tickangle=-35)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            counts = df["sentiment"].value_counts().reset_index()
            counts.columns = ["sentiment", "count"]
            fig = px.pie(counts, names="sentiment", values="count", title="Sentiment Distribution")
            st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)

        with col3:
            counts = df["resolution"].value_counts().reset_index()
            counts.columns = ["resolution", "count"]
            fig = px.pie(counts, names="resolution", values="count", title="Resolution Distribution")
            st.plotly_chart(fig, use_container_width=True)

        with col4:
            counts = df["escalation"].value_counts().reset_index()
            counts.columns = ["escalation", "count"]
            fig = px.bar(counts, x="escalation", y="count", title="Escalation Distribution")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Recent analyzed conversations")
        visible = [
            "customer_message", "agent_response", "language", "intent",
            "sentiment", "resolution", "escalation",
            "escalation_probability", "priority", "confidence",
            "status", "created_at"
        ]
        st.dataframe(
            df[visible],
            use_container_width=True,
            hide_index=True,
        )
