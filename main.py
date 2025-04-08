"""
このファイルは、Webアプリのメイン処理が記述されたファイルです。
"""

############################################################
# 1. ライブラリの読み込み
############################################################
import pandas as pd 
# .env」ファイルから環境変数を読み込むための関数
from dotenv import load_dotenv
# ログ出力を行うためのモジュール
import logging
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
    page_title=ct.APP_NAME
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
    st.exception(e)
    # エラーメッセージの画面表示
    st.error(build_error_message(ct.INITIALIZE_ERROR_MESSAGE), icon=ct.ERROR_ICON)
    # 後続の処理を中断
    st.stop()

# アプリ起動時のログファイルへの出力
if not "initialized" in st.session_state:
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
    st.write("#### 開発者モード")
    # 開発者モードのトグル
    debug_mode = st.toggle(
        "デバッグログを表示",
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
    st.exception(e)
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
    content = ""  # 追加: contentを初期化

    # ==========================================
    # 7-0. ファイル更新チェック　新設
    # ==========================================
    # ファイル更新チェック
    if "upload_time" in st.session_state and "uploaded_files" in st.session_state:
        file_paths = [file.name for file in st.session_state.uploaded_files]
        ref_time = st.session_state.upload_time
        files_updated = check_files_for_updates(file_paths, ref_time)
    else:
        files_updated = []
    if files_updated:
        st.info(ct.FILE_UPDATE_MESSAGE, icon=ct.FILE_UPDATE_ICON)

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
    
    # 「st.spinner」でグルグル回っている間、表示の不具合が発生しないよう空のエリアを表示
    res_box = st.empty()
    # LLMによる回答生成（回答生成が完了するまでグルグル回す）
    with st.spinner(ct.SPINNER_TEXT):
        try:
            # 画面読み込み時に作成したRetrieverを使い、Chainを実行
            llm_response = utils.get_llm_response(chat_message)
            if "answer" not in llm_response:  # 追加: エラーチェック
                raise ValueError("LLMの回答が取得できませんでした。")
        except Exception as e:
            # エラーログの出力
            logger.error(f"{ct.GET_LLM_RESPONSE_ERROR_MESSAGE}\n{e}")
            st.exception(e)
            # エラーメッセージの画面表示
            st.error(build_error_message(ct.GET_LLM_RESPONSE_ERROR_MESSAGE), icon=ct.ERROR_ICON)
            # 後続の処理を中断
            st.stop()

    # ==========================================
    # 7-3. LLMからの回答表示
    # ==========================================
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

                # 追加: 空の回答に対するフォールバックメッセージ
                if not content:
                    content = "申し訳ありませんが、回答を生成できませんでした。"

            except Exception as e:
                # エラーログの出力
                logger.error(f"{ct.DISP_ANSWER_ERROR_MESSAGE}\n{e}")
                st.exception(e)
                # エラーメッセージの画面表示
                st.error(build_error_message(ct.DISP_ANSWER_ERROR_MESSAGE), icon=ct.ERROR_ICON)
                # 後続の処理を中断
                st.stop()

    # ==========================================
    # DEBUGログの表示（開発者モードON時のみ）
    # ==========================================
    if st.session_state.get("debug_mode", False):
        with st.expander("DEBUGログ", expanded=True):
            st.markdown("### LLMレスポンス（生データ）")
            st.json(llm_response)
            
            st.markdown("### ログファイル内容")
            try:
                with open("logs/application.log", "r", encoding="utf-8") as f:
                    log_content = f.read()
                    st.code(log_content[-5000:] if len(log_content) > 5000 else log_content, language="text")
            except FileNotFoundError:
                st.warning("ログファイルが見つかりませんでした。")

    # ==========================================
    # 7-4. 会話ログへの追加
    # ==========================================
    # 表示用の会話ログにユーザーメッセージを追加
    st.session_state.messages.append({"role": "user", "content": chat_message})
    # 表示用の会話ログにAIメッセージを追加
    st.session_state.messages.append({"role": "assistant", "content": content})