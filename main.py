"""
このファイルは、Webアプリのメイン処理が記述されたファイルです。
"""

############################################################
# 1. ライブラリの読み込み
############################################################
import os
import logging
from datetime import datetime
# 「.env」ファイルから環境変数を読み込むための関数
from dotenv import load_dotenv
# streamlitアプリの表示を担当するモジュール
import streamlit as st
# （自作）画面表示以外の様々な関数が定義されているモジュール
import utils
from utils import check_files_for_updates, build_error_message
# （自作）アプリ起動時に実行される初期化処理が記述された関数
from initialize import initialize
# （自作）画面表示系の関数が定義されているモジュール
import components as cn
# （自作）変数（定数）がまとめて定義・管理されているモジュール
import constants as ct


############################################################
# 2. 設定関連
############################################################
# ブラウザタブの表示文言を設定
st.set_page_config(
    page_title=ct.APP_NAME,
    layout="wide"
)

# ログ出力を行うためのロガーの設定
logger = logging.getLogger(ct.LOGGER_NAME)


############################################################
# 3. 初期化処理
############################################################
try:
    # 初期化処理（「initialize.py」の「initialize」関数を実行）
    initialize()
except Exception as e:
    # エラーログの出力
    logger.error(f"{ct.INITIALIZE_ERROR_MESSAGE}\n{e}")
    # エラーメッセージの画面表示
    st.error(build_error_message(ct.INITIALIZE_ERROR_MESSAGE), icon=ct.ERROR_ICON)
    # 後続の処理を中断
    st.stop()

# アプリ起動時のログファイルへの出力
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    logger.info(ct.APP_BOOT_MESSAGE)


############################################################
# 4. 初期表示
############################################################
# タイトル表示
cn.display_app_title()

# サイドバー表示
cn.display_sidebar()

# 開発者モードのトグル（サイドバーの最後に追加）
with st.sidebar:
    st.divider()
    st.write("#### 開発者設定")
    # 開発者モードのトグル
    debug_mode = st.toggle(
        "デバッグ情報を表示",
        value=st.session_state.get("debug_mode", False),
        key="debug_toggle"
    )
    # トグルの状態をセッション変数に保存
    st.session_state.debug_mode = debug_mode

# AIメッセージの初期表示
cn.display_initial_ai_message()


############################################################
# 5. 会話ログの表示
############################################################
try:
    # 会話ログの表示
    cn.display_conversation_log()
except Exception as e:
    # エラーログの出力
    logger.error(f"{ct.CONVERSATION_LOG_ERROR_MESSAGE}\n{e}")
    # エラーメッセージの画面表示
    st.error(build_error_message(ct.CONVERSATION_LOG_ERROR_MESSAGE), icon=ct.ERROR_ICON)
    # 後続の処理を中断
    st.stop()


############################################################
# 6. チャット入力の受け付け
############################################################
chat_message = st.chat_input(ct.CHAT_INPUT_HELPER_TEXT)


############################################################
# 7. チャット送信時の処理
############################################################
if chat_message:
    # ==========================================
    # 7-0. ファイル更新チェック
    # ==========================================
    try:
        # ファイル更新チェック
        if "upload_time" in st.session_state and "uploaded_files" in st.session_state:
            file_paths = [file.name for file in st.session_state.uploaded_files]
            ref_time = st.session_state.upload_time
            files_updated = check_files_for_updates(file_paths, ref_time)
            
            # 更新があった場合の処理
            if files_updated:
                st.info(ct.FILE_UPDATE_MESSAGE, icon=ct.FILE_UPDATE_ICON)
                logger.info(f"ファイル更新を検知: {', '.join(files_updated)}")
    except Exception as e:
        logger.warning(f"ファイル更新チェックエラー: {e}")

    # ==========================================
    # 7-1. ユーザーメッセージの表示
    # ==========================================
    # ユーザーメッセージのログ出力
    logger.info({"message": chat_message, "application_mode": st.session_state.mode})

    # ユーザーメッセージを表示
    with st.chat_message("user"):
        st.markdown(chat_message)

    # ==========================================
    # 7-2. LLMからの回答取得
    # ==========================================
    # 回答生成中のメッセージを先に表示
    answer_box = st.empty()
    with answer_box.container():
        with st.chat_message("assistant"):
            st.markdown("回答生成中...")
    
    # LLMによる回答生成（回答生成が完了するまでグルグル回す）
    with st.spinner(ct.SPINNER_TEXT):
        try:
            # 画面読み込み時に作成したRetrieverを使い、Chainを実行
            llm_response = utils.get_llm_response(chat_message)
            
            # レスポンスの検証
            if not utils.validate_llm_response(llm_response):
                raise ValueError("無効なLLMレスポンスを受信しました")
                
        except Exception as e:
            # エラーログの出力
            logger.error(f"{ct.GET_LLM_RESPONSE_ERROR_MESSAGE}\n{e}")
            # エラーメッセージの画面表示
            st.error(build_error_message(ct.GET_LLM_RESPONSE_ERROR_MESSAGE), icon=ct.ERROR_ICON)
            # 表示用の会話ログにユーザーメッセージを追加
            st.session_state.messages.append({"role": "user", "content": chat_message})
            # 後続の処理を中断
            st.stop()

    # ==========================================
    # 7-3. LLMからの回答表示
    # ==========================================
    content = None  # 表示用のコンテンツを格納する変数
    
    with answer_box.container():
        with st.chat_message("assistant"):
            try:
                # ==========================================
                # モードが「社内文書検索」の場合
                # ==========================================
                if st.session_state.mode == ct.ANSWER_MODE_1:
                    # 入力内容と関連性が高い社内文書のありかを表示
                    content = cn.display_search_llm_response(llm_response)

                # ==========================================
                # モードが「社内問い合わせ」の場合
                # ==========================================
                elif st.session_state.mode == ct.ANSWER_MODE_2:
                    # 入力に対しての回答と、参照した文書のありかを表示
                    content = cn.display_contact_llm_response(llm_response)
                
                # AIメッセージのログ出力
                logger.info({"message": content, "application_mode": st.session_state.mode})

            except Exception as e:
                # エラーログの出力
                logger.error(f"{ct.DISP_ANSWER_ERROR_MESSAGE}\n{e}")
                # エラーメッセージの画面表示
                st.error(build_error_message(ct.DISP_ANSWER_ERROR_MESSAGE), icon=ct.ERROR_ICON)
                # 最低限の内容を持つコンテンツの作成
                error_message = "回答の表示中にエラーが発生しました。もう一度お試しください。"
                content = {
                    "mode": st.session_state.mode,
                    "answer": error_message
                }

    # ==========================================
    # 7-4. 会話ログへの追加
    # ==========================================
    # コンテンツが生成されている場合のみ追加処理を実行
    if content:
        # 表示用の会話ログにユーザーメッセージを追加
        st.session_state.messages.append({"role": "user", "content": chat_message})
        # 表示用の会話ログにAIメッセージを追加
        st.session_state.messages.append({"role": "assistant", "content": content})
    else:
        # コンテンツが生成されていない場合のエラー処理
        logger.error("コンテンツが生成されませんでした")
        st.error("応答の処理中に問題が発生しました。もう一度お試しください。", icon=ct.ERROR_ICON)