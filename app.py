import streamlit as st
import asyncio  # For asynchronous operations


async def main():
    st.title("PromptIDE GUI")  # Add a title to your app
    st.write("Welcome to the PromptIDE GUI!")  # Display some text


if __name__ == "__main__":
    asyncio.run(main())