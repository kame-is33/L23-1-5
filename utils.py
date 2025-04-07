"""
このファイルは、画面表示以外の様々な関数定義のファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
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


############################################################
# 設定関連
############################################################
# 「.env」ファイルで定義した環境変数の読み込み
load_dotenv()


############################################################
# 関数定義
############################################################

def get_source_icon(source):
    """
    メッセージと一緒に表示するアイコンの種類を取得

    Args:
        source: 参照元のありか

    Returns:
        メッセージと一緒に表示するアイコンの種類
    """
    # 参照元がWebページの場合とファイルの場合で、取得するアイコンの種類を変える
    if source.startswith("http"):
        icon = ct.LINK_SOURCE_ICON
    else:
        icon = ct.DOC_SOURCE_ICON
    
    return icon


def build_error_message(message):
    """
    エラーメッセージと管理者問い合わせテンプレートの連結

    Args:
        message: 画面上に表示するエラーメッセージ

    Returns:
        エラーメッセージと管理者問い合わせテンプレートの連結テキスト
    """
    return "\n".join([message, ct.COMMON_ERROR_MESSAGE])


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

    # 会話履歴なしでもLLMに理解してもらえる、独立した入力テキストを取得するためのプロンプトテンプレートを作成
    question_generator_template = ct.SYSTEM_PROMPT_CREATE_INDEPENDENT_TEXT
    question_generator_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", question_generator_template),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    # 社員情報関連のキーワードを検出
    employee_keywords = ["人事", "従業員", "社員", "部署", "スキル"]
    is_employee_query = any(keyword in chat_message for keyword in employee_keywords)
    
    # 社員情報を含む質問の場合、CSVデータを直接提供
    employee_context = ""
    if is_employee_query and st.session_state.mode == ct.ANSWER_MODE_2:
        try:
            csv_path = "./data/社員について/社員名簿.csv"
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                
                # 列名の表記揺れ対策
                if "所属部署" not in df.columns and "部署" in df.columns:
                    df = df.rename(columns={"部署": "所属部署"})
                
                # 「人事部」のフィルタリング（必要な場合）
                if "人事部" in chat_message:
                    if "所属部署" in df.columns:
                        filtered_df = df[df["所属部署"].str.contains("人事", na=False)]
                        if not filtered_df.empty:
                            employee_context = filtered_df.to_markdown(index=False)
                    else:
                        employee_context = df.to_markdown(index=False)
                else:
                    # 社員情報全体を提供
                    employee_context = df.to_markdown(index=False)
                
                # 社員情報をプロンプトに追加
                if employee_context:
                    chat_message = f"以下の社員情報を参照して質問に答えてください：\n\n{employee_context}\n\n質問: {chat_message}"
        except Exception as e:
            logging.getLogger(ct.LOGGER_NAME).warning(f"社員情報の読み込みに失敗しました: {e}")

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
    
    # 社員情報関連の質問で、社内問い合わせモードの場合、取得した社員情報を追加
    if is_employee_query and st.session_state.mode == ct.ANSWER_MODE_2 and employee_context:
        # 社員情報が提供されている場合、結果が空でも回答を生成
        if llm_response["answer"] == ct.INQUIRY_NO_MATCH_ANSWER:
            # 直接LLMに質問して回答を取得
            direct_response = llm.invoke(f"以下の社員情報を参照して質問に答えてください：\n\n{employee_context}\n\n質問: {chat_message}")
            llm_response["answer"] = direct_response.content
            
            # 社員情報のCSVを参照元として追加
            if "context" not in llm_response:
                from langchain_core.documents import Document
                llm_response["context"] = [
                    Document(page_content="社員情報", metadata={"source": "./data/社員について/社員名簿.csv"})
                ]
    
    # LLMレスポンスを会話履歴に追加
    st.session_state.chat_history.extend([HumanMessage(content=chat_message), llm_response["answer"]])

    return llm_response


def check_files_for_updates():
    """
    ファイルの更新を検知し、必要に応じてベクターストアを更新する
    
    Returns:
        bool: 更新があった場合はTrue、なければFalse
    """
    updated = False
    logger = logging.getLogger(ct.LOGGER_NAME)
    
    # 初回実行時はファイルメタデータを初期化
    if "file_metadata" not in st.session_state:
        st.session_state.file_metadata = {}
    
    # すべてのファイルを再帰的にチェック
    for root, dirs, files in os.walk(ct.RAG_TOP_FOLDER_PATH):
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file_path)[1]
            
            # サポート対象のファイル形式のみチェック
            if file_ext in ct.SUPPORTED_EXTENSIONS:
                # 現在のファイルメタデータを取得
                try:
                    stat_info = os.stat(file_path)
                    current_metadata = {
                        "size": stat_info.st_size,
                        "mtime": stat_info.st_mtime
                    }
                    
                    # 前回のメタデータと比較
                    if (file_path in st.session_state.file_metadata and 
                        (st.session_state.file_metadata[file_path]["size"] != current_metadata["size"] or 
                         st.session_state.file_metadata[file_path]["mtime"] != current_metadata["mtime"])):
                        logger.info(f"ファイルが更新されました: {file_path}")
                        updated = True
                    
                    # メタデータを更新
                    st.session_state.file_metadata[file_path] = current_metadata
                    
                except Exception as e:
                    logger.error(f"ファイルメタデータの取得に失敗: {file_path}\n{e}")
    
    # 更新があった場合はベクターストアを再構築
    if updated and "retriever" in st.session_state:
        logger.info("ファイル更新を検知したため、ベクターストアを再構築します")
        # session_stateからretrieverを削除して再初期化を強制
        del st.session_state.retriever
        # initialize.pyのinitialize_retriever関数を再度呼び出す
        from initialize import initialize_retriever
        initialize_retriever()
        return True
    
    return False