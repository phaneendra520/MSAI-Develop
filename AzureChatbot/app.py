"""Gradio web interface for the Azure-integrated chatbot."""
import gradio as gr

from chatbot import Chatbot

bot = Chatbot()
status = "connected" if bot.azure.available else "NOT configured (sentiment features disabled)"
print(f"Azure AI Language: {status}")

demo = gr.ChatInterface(
    fn=lambda message, history: bot.reply(message),
    title="Campus Help Desk Bot + Azure AI Language",
    description=f"Traditional intent chatbot with Azure sentiment analysis. Azure status: {status}. "
                "Type 'help' to see what I can do.",
    examples=["help", "I forgot my password and I'm really annoyed",
              "analyze: The new library hours are fantastic!", "asdfgh qwerty"],
)

if __name__ == "__main__":
    demo.launch(share=True)
