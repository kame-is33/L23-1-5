"""
このファイルは、画面表示以外の様々な関数定義のファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
import pandas as pd
import os
import logging
import streamlit as st
import time
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, Document
from langchain_openai import ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import constants as ct


############################################################
# 関数定義
############################################################

def build_error_message(message):
    """
    エラーメッセージと管理者問い合わせテンプレートの連結

    Args:
        message: 画面上に表示するエラーメッセージ

    Returns:
        エラーメッセージと管理者問い合わせテンプレートの連結テキスト
    """
    return "\n".join([message, ct.COMMON_ERROR_MESSAGE])


def get_source_icon(file_path):
    """
    参照元のファイルパスに応じて、適切なアイコンを取得する

    Args:
        file_path: 参照元のファイルパス

    Returns:
        適切なアイコン
    """
    # URLの特徴を検出
    if file_path.startswith("http") or file_path.startswith("www"):
        return ct.LINK_SOURCE_ICON
    else:
        return ct.DOC_SOURCE_ICON


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
    logger = logging.getLogger(ct.LOGGER_NAME)

    # 社員情報関連のキーワードを検出
    is_employee_query = any(keyword in chat_message for keyword in ct.EMPLOYEE_KEYWORDS)
    
    # ベクターストアを取得
    vectorstore = st.session_state.vectorstore
    
    # クエリに応じてRetrieverのドキュメント取得数を調整
    if is_employee_query:
        # 社員情報クエリの場合は取得数を増加
        retriever = vectorstore.as_retriever(search_kwargs={"k": ct.RETRIEVER_DOCUMENT_COUNT_EMPLOYEE})
        logger.info(f"社員情報に関する質問を検出しました。ドキュメント取得数を{ct.RETRIEVER_DOCUMENT_COUNT_EMPLOYEE}に設定します。")
    else:
        # 通常のクエリの場合はデフォルト値を使用
        retriever = vectorstore.as_retriever(search_kwargs={"k": ct.RETRIEVER_DOCUMENT_COUNT_DEFAULT})
    
    # 社内問い合わせモードで社員情報に関する質問の場合、専用処理ルートを実行
    if is_employee_query and st.session_state.mode == ct.ANSWER_MODE_2 and "employee_csv_path" in st.session_state:
        try:
            # 検出済みの社員名簿ファイルパスを取得
            employee_csv_path = st.session_state.employee_csv_path
            logger.info(f"社員情報に関する質問を検出しました。社員名簿データを処理します: {employee_csv_path}")
            
            # 社員名簿データの読み込み
            employee_df = pd.read_csv(employee_csv_path)
            
            # CSVファイルの構造を分析して重要なカラムを特定
            columns_info = analyze_csv_structure(employee_csv_path)
            
            # 社員データを文字列に変換（表形式を維持）
            employee_data = format_employee_data(employee_df)
            
            # 社員情報専用のプロンプトを作成
            employee_prompt = ChatPromptTemplate.from_messages([
                ("system", ct.SYSTEM_PROMPT_EMPLOYEE.format(employee_data=employee_data)),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ])
            
            # 社員情報専用のチェーンを作成
            employee_chain = employee_prompt | llm
            
            # 社員情報専用のチェーンを実行
            response = employee_chain.invoke({
                "input": chat_message,
                "chat_history": st.session_state.chat_history
            })
            
            # 応答結果を整形
            llm_response = {
                "answer": response.content,
                "context": [
                    Document(
                        page_content=employee_data,
                        metadata={"source": employee_csv_path, "is_employee_data": True}
                    )
                ]
            }
            
            # LLMレスポンスを会話履歴に追加
            st.session_state.chat_history.extend([
                HumanMessage(content=chat_message), 
                HumanMessage(content=response.content)
            ])
            
            logger.info("社員情報専用処理ルートで回答を生成しました。")
            return llm_response
                
        except Exception as e:
            logger.error(f"社員情報の処理中にエラーが発生しました: {e}")
            # エラーが発生した場合は通常の処理フローに戻る
    
list()
                
                # フォールバック回答の生成
                fallback_answer = f"""
                ### 社員情報検索結果
                
                社員名簿には以下の情報が含まれています：
                
                {', '.join(columns)}
                
                より具体的な質問をいただくと、詳細な情報を提供できます。
                """
                
                llm_response["answer"] = fallback_answer
                
                # コンテキストに社員名簿の情報を追加
                llm_response["context"] = [
                    Document(
                        page_content="社員名簿の概要情報",
                        metadata={"source": ct.EMPLOYEE_CSV_PATH}
                    )
                ]
                
                logger.info("社員情報に関するフォールバック回答を生成しました。")
        except Exception as e:
            logger.error(f"社員情報のフォールバック処理中にエラーが発生しました: {e}")

    # LLMレスポンスを会話履歴に追加
    st.session_state.chat_history.extend([
        HumanMessage(content=chat_message), 
        HumanMessage(content=llm_response["answer"])
    ])

    return llm_response


def analyze_csv_structure(csv_path):
    """
    CSVファイルの構造を分析し、重要なカラム名と統計情報を取得する
    
    Args:
        csv_path: CSVファイルのパス
        
    Returns:
        CSVファイルの構造情報を含む辞書
    """
    try:
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            
            # CSVファイルの基本情報を収集
            info = {
                "columns": df.columns.tolist(),
                "row_count": len(df),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "sample_data": df.head(2).to_dict(orient="records"),
                "unique_values": {
                    col: df[col].unique().tolist() 
                    for col in df.columns 
                    if df[col].nunique() < 10 and pd.api.types.is_object_dtype(df[col])
                }
            }
            
            # 部署カラムの特定
            dept_columns = [col for col in df.columns if any(keyword in col for keyword in ["部署", "所属"])]
            if dept_columns:
                info["dept_column"] = dept_columns[0]
                info["departments"] = df[dept_columns[0]].unique().tolist()
            
            # スキルカラムの特定
            skill_columns = [col for col in df.columns if any(keyword in col for keyword in ["スキル", "能力", "技術"])]
            if skill_columns:
                info["skill_column"] = skill_columns[0]
            
            return info
        else:
            return None
    except Exception as e:
        logging.getLogger(ct.LOGGER_NAME).error(f"CSV構造分析エラー: {e}")
        return None


def format_employee_data(df):
    """
    社員データを見やすい形式にフォーマット
    
    Args:
        df: 社員データのDataFrame
        
    Returns:
        フォーマットされた社員データの文字列
    """
    # 行数が少ない場合は表形式で全データを表示
    if len(df) <= 20:
        return df.to_string(index=False)
    
    # 行数が多い場合は概要情報と部署別の集計を表示
    header = f"社員データ概要（全{len(df)}名）\n\n"
    
    # 部署カラムの特定
    dept_col = None
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ["部署", "所属"]):
            dept_col = col
            break
    
    # 部署別の人数集計
    dept_summary = ""
    if dept_col:
        dept_counts = df[dept_col].value_counts().reset_index()
        dept_counts.columns = [dept_col, "人数"]
        dept_summary = f"\n部署別人数:\n{dept_counts.to_string(index=False)}\n\n"
    
    # 生のデータテーブル
    data_table = "社員データ（全員）:\n" + df.to_string(index=False)
    
    return header + dept_summary + data_table


def check_files_for_updates():
    """
    ファイルの更新を検知する関数
    
    Returns:
        更新があったかどうかのブール値
    """
    # 本来はファイルの更新日時を比較するなど実装が必要だが、ここではサンプル実装
    return False