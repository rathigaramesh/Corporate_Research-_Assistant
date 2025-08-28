import gradio as gr
import time
from agents import agent_instance  # your initialized LangChain agent


class GradioApp:
    def __init__(self):
        self.agent = agent_instance
        self.chat_history = []

    def process_query(self, message, history):
        """Process user query and return response"""
        try:
            # Add user message
            self.chat_history.append(("👤 You", message))

            # Simulate "thinking" delay
            time.sleep(0.4)

            # Query agent
            response = self.agent.query_agent(message)

            # Add agent reply
            self.chat_history.append(("🤖 Responsing...", response))

            return self.chat_history
        except Exception as e:
            error_msg = f"⚠️ Sorry, I encountered an error: {str(e)}"
            self.chat_history.append(("🤖 Assistant", error_msg))
            return self.chat_history

    def clear_chat(self):
        """Clear chat history"""
        self.chat_history = []
        return self.chat_history, ""

    def create_interface(self):
        """Create the Gradio interface"""
        with gr.Blocks(
            title="Corporate Research Assistant",
            theme=gr.themes.Soft(),
             css="""
            .status-label {
                background-color: #16a34a !important;  /* ✅ green-600 */
                color: white !important;
                font-weight: bold !important;
                border-radius: 8px !important;
                padding: 8px !important;
                text-align: center !important;
            }
            """
        ) as demo:
            # Header
            with gr.Row():
                gr.Markdown(
                    """
                    # 🏢 Corporate Research Assistant  
                    Ask me about **HR policies**, **technical documentation**, or **current events**!  
                    """,
                )

            # Status Dashboard
            with gr.Row():
                with gr.Column():
                    gr.Markdown("## ⚡ System Status")
                    with gr.Row():
                        agent_status = gr.Label("✅ Agent Ready", label="Agent", elem_classes="status-label")
                        vectordb_status = gr.Label("📚 VectorDB Connected", label="Knowledge Base", elem_classes="status-label")
                        llm_status = gr.Label("🧠 Model Running", label="LLM", elem_classes="status-label")
            # Chat UI
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        label="💬 Conversation",
                        height=500,
                        bubble_full_width=False,
                        show_label=False
                    )

                    with gr.Row():
                        msg = gr.Textbox(
                            label="Type your question",
                            placeholder="e.g. What is the leave policy?",
                            lines=2,
                            scale=4
                        )
                        submit_btn = gr.Button("🚀 Send", variant="primary")

                    clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")

                # Sidebar info
                with gr.Column(scale=1):
                    gr.Markdown("### 📖 Knowledge Bases")
                    gr.Markdown(
                        """
                        - **HR Policies**: Employee handbook, benefits, PTO  
                        - **Technical Docs**: APIs, deployment guides, code standards  
                        - **Web Search**: Current events, market trends  
                        """
                    )

            # Event handlers
            msg.submit(
                self.process_query,
                [msg, chatbot],
                [chatbot],
            ).then(
                lambda: "", None, msg
            )

            submit_btn.click(
                self.process_query,
                [msg, chatbot],
                [chatbot],
            ).then(
                lambda: "", None, msg
            )

            clear_btn.click(
                self.clear_chat,
                None,
                [chatbot, msg],
            )

        return demo


def main():
    """Main function to run the application"""
    print("🚀 Starting Corporate Research Assistant...")

    app = GradioApp()
    demo = app.create_interface()

    demo.launch(
        server_name="localhost",
        server_port=7860,
        share=False,
    )


if __name__ == "__main__":
    main()