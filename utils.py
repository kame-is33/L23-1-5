import pandas as pd  # この行を追加して pandas をインポート
import os
import logging
from dotenv import load_dotenv
import streamlit as st
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import constants as ct

def get_llm_response(chat_message):
    """
    LLMからの回答取得

    Args:
        chat_message: ユーザー入力値

    Returns:
        LLMからの回答
    """
    # LLMのオブジェクトを用意
    llm = ChatOpenAI(model_name=ct.MODEL, temperature=ct.TEMPERATURE)

    # 社員情報関連のキーワードを検出
    employee_keywords = ["人事", "従業員", "社員", "部署", "スキル"]
    is_employee_query = any(keyword in chat_message for keyword in employee_keywords)
    
    # 社員情報の追加処理
    if is_employee_query and st.session_state.mode == ct.ANSWER_MODE_2:
        try:
            csv_path = "./data/社員について/社員名簿.csv"
            if os.path.exists(csv_path):
                # 社員データの読み込み
                employee_df = pd.read_csv(csv_path)
                
                # 人事部のフィルタリング
                dept_column = None
                for col in employee_df.columns:
                    if "部署" in col or "所属" in col:
                        dept_column = col
                        break
                
                # 強制的に回答を生成
                from langchain_core.documents import Document
                dummy_doc = Document(
                    page_content="社員情報があります",
                    metadata={"source": "./data/社員について/社員名簿.csv"}
                )
                
                # チャットメッセージをモディファイ
                chat_message = f"社員名簿を参照して次の質問に答えてください: {chat_message}"
        except Exception as e:
            logging.getLogger(ct.LOGGER_NAME).warning(f"社員情報の読み込みに失敗しました: {e}")

    # 会話履歴なしでもLLMに理解してもらえる、独立した入力テキストを取得するためのプロンプトテンプレートを作成
    question_generator_template = ct.SYSTEM_PROMPT_CREATE_INDEPENDENT_TEXT
    question_generator_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", question_generator_template),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    # モードによってLLMから回答を取得する用のプロンプトを変更
    if st.session_state.mode == ct.ANSWER_MODE_1:
        # モードが「社内文書検索」の場合のプロンプト
        question_answer_template = ct.SYSTEM_PROMPT_DOC_SEARCH
    else:
        # モードが「社内問い合わせ」の場合のプロンプト
        question_answer_template = ct.SYSTEM_PROMPT_INQUIRY

    # LLMから回答を取得する用のプロンプトテンプレートを作成
    question_answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", question_answer_template),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    # 会話履歴なしでもLLMに理解してもらえる、独立した入力テキストを取得するためのRetrieverを作成
    history_aware_retriever = create_history_aware_retriever(
        llm, st.session_state.retriever, question_generator_prompt
    )

    # LLMから回答を取得する用のChainを作成
    question_answer_chain = create_stuff_documents_chain(llm, question_answer_prompt)
    # 「RAG x 会話履歴の記憶機能」を実現するためのChainを作成
    chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # LLMへのリクエストとレスポンス取得
    llm_response = chain.invoke({"input": chat_message, "chat_history": st.session_state.chat_history})

    # 社員情報関連の質問で「情報が見つからなかった」場合は直接問い合わせ
    if is_employee_query and llm_response["answer"] == ct.INQUIRY_NO_MATCH_ANSWER:
        try:
            csv_path = "./data/社員について/社員名簿.csv"
            if os.path.exists(csv_path):
                # 社員データの読み込み
                employee_df = pd.read_csv(csv_path)
                
                # 部署列の特定
                dept_column = None
                for col in employee_df.columns:
                    if "部署" in col or "所属" in col:
                        dept_column = col
                        break
                
                # 人事部のフィルタリング（部署列が見つかった場合）
                if dept_column and "人事" in chat_message:
                    filtered_df = employee_df[employee_df[dept_column].str.contains("人事", na=False)]
                else:
                    filtered_df = employee_df
                
                if not filtered_df.empty:
                    # 社員情報をマークダウンテーブルに変換
                    employee_table = filtered_df.to_markdown(index=False)
                    
                    # 直接LLMに社員情報と質問を送信
                    direct_prompt = f"""
                    以下の社員情報を使って質問に答えてください。
                    
                    {employee_table}
                    
                    質問: {chat_message}
                    """
                    
                    direct_response = llm.invoke(direct_prompt)
                    
                    # レスポンスの上書き
                    llm_response["answer"] = direct_response.content
                    
                    # context がない場合は作成
                    if "context" not in llm_response or not llm_response["context"]:
                        from langchain_core.documents import Document
                        llm_response["context"] = [Document(
                            page_content="社員名簿の情報",
                            metadata={"source": "./data/社員について/社員名簿.csv"}
                        )]
        except Exception as e:
            logging.getLogger(ct.LOGGER_NAME).warning(f"社員情報による直接回答の生成に失敗しました: {e}")

    # LLMレスポンスを会話履歴に追加
    st.session_state.chat_history.extend([HumanMessage(content=chat_message), llm_response["answer"]])

    return llm_response