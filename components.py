"""
このファイルは、画面表示に特化した関数定義のファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
import streamlit as st
import logging
import utils
import constants as ct


############################################################
# 関数定義
############################################################

def display_app_title():
    """
    タイトル表示
    """
    st.markdown(f"## {ct.APP_NAME}")


def display_sidebar():
    """
    サイドバーの表示
    """
    # 回答モードのラジオボタンを表示
    st.sidebar.markdown("### 利用目的")
    st.session_state.mode = st.sidebar.radio(
        label="",
        options=[ct.ANSWER_MODE_1, ct.ANSWER_MODE_2],
        label_visibility="collapsed"
    )
    st.sidebar.divider()
    
    # 「社内文書検索」の機能説明
    st.sidebar.markdown(ct.SIDEBAR_SEARCH_TITLE)
    st.sidebar.info(ct.SIDEBAR_SEARCH_DESCRIPTION)
    st.sidebar.code(f"{ct.EXAMPLE_TITLE}\n{ct.SIDEBAR_SEARCH_EXAMPLE}", wrap_lines=True, language=None)

    # 「社内問い合わせ」の機能説明
    st.sidebar.markdown(ct.SIDEBAR_INQUIRY_TITLE)
    st.sidebar.info(ct.SIDEBAR_INQUIRY_DESCRIPTION)
    st.sidebar.code(f"{ct.EXAMPLE_TITLE}\n{ct.SIDEBAR_INQUIRY_EXAMPLE}", wrap_lines=True, language=None)

    # 社員情報に関する説明
    st.sidebar.markdown(ct.SIDEBAR_EMPLOYEE_TITLE)
    st.sidebar.info(ct.SIDEBAR_EMPLOYEE_DESCRIPTION)
    st.sidebar.code(f"{ct.EXAMPLE_TITLE}\n{ct.SIDEBAR_EMPLOYEE_EXAMPLE}", wrap_lines=True, language=None)


def display_initial_ai_message():
    """
    AIメッセージの初期表示
    """
    with st.chat_message("assistant"):
        # 「st.success()」とすると緑枠で表示される
        st.success("こんにちは。私は社内文書の情報をもとに回答する生成AIチャットボットです。サイドバーで利用目的を選択し、画面下部のチャット欄からメッセージを送信してください。")
        st.warning("具体的に入力したほうが期待通りの回答を得やすいです。", icon=ct.WARNING_ICON)


def display_conversation_log():
    """
    会話ログの一覧表示
    """
    # 会話ログのループ処理
    for message in st.session_state.messages:
        # 「message」辞書の中の「role」キーには「user」か「assistant」が入っている
        with st.chat_message(message["role"]):

            # ユーザー入力値の場合、そのままテキストを表示するだけ
            if message["role"] == "user":
                st.markdown(message["content"])
            
            # LLMからの回答の場合
            else:
                # 「社内文書検索」の場合、テキストの種類に応じて表示形式を分岐処理
                if message["content"]["mode"] == ct.ANSWER_MODE_1:
                    
                    # ファイルのありかの情報が取得できた場合（通常時）の表示処理
                    if not "no_file_path_flg" in message["content"]:
                        # ==========================================
                        # ユーザー入力値と最も関連性が高いメインドキュメントのありかを表示
                        # ==========================================
                        # 補足文の表示
                        st.markdown(message["content"]["main_message"])

                        # メインドキュメントのアイコンと情報を表示
                        icon = utils.get_source_icon(message['content']['main_file_path'])
                        main_info = message['content']['main_file_path']
                        if "main_page_number" in message["content"]:
                            main_info += f"（Page #{message['content']['main_page_number']}）"
                        st.success(main_info, icon=icon)
                        
                        # ==========================================
                        # ユーザー入力値と関連性が高いサブドキュメントのありかを表示
                        # ==========================================
                        if "sub_message" in message["content"] and "sub_choices" in message["content"]:
                            # 補足メッセージの表示
                            st.markdown("##### 関連資料")
                            st.markdown(message["content"]["sub_message"])

                            # サブドキュメントのありかを一覧表示
                            for sub in message["content"]["sub_choices"]:
                                sub_text = sub['source']
                                if sub.get("page_number"):
                                    sub_text += f"（Page #{sub['page_number']}）"
                                icon = utils.get_source_icon(sub['source'])
                                st.info(sub_text, icon=icon)
                    # ファイルのありかの情報が取得できなかった場合、LLMからの回答のみ表示
                    else:
                        st.markdown(message["content"]["answer"])
                
                # 「社内問い合わせ」の場合の表示処理
                else:
                    # LLMからの回答を表示
                    st.markdown(message["content"]["answer"])

                    # 参照元のありかを一覧表示
                    if "file_info_list" in message["content"]:
                        # 区切り線の表示
                        st.divider()
                        # 「情報源」の文字を太字で表示
                        st.markdown(f"##### {message['content']['message']}")
                        # ドキュメントのありかを一覧表示
                        for file_info in message["content"]["file_info_list"]:
                            # 参照元のありかに応じて、適したアイコンを取得
                            icon = utils.get_source_icon(file_info)
                            st.info(file_info, icon=icon)


def display_search_llm_response(llm_response):
    """
    「社内文書検索」モードにおけるLLMレスポンスを表示

    Args:
        llm_response: LLMからの回答

    Returns:
        LLMからの回答を画面表示用に整形した辞書データ
    """
    logger = logging.getLogger(ct.LOGGER_NAME)
    
    # LLMからのレスポンスに参照元情報が入っており、かつ「該当資料なし」が回答として返された場合
    if "context" in llm_response and llm_response["context"] and llm_response["answer"] != ct.NO_DOC_MATCH_ANSWER:
        try:
            # ==========================================
            # ユーザー入力値と最も関連性が高いメインドキュメントのありかを表示
            # ==========================================
            # LLMからのレスポンス（辞書）の「context」属性の中の「0」に、最も関連性が高いドキュメント情報が入っている
            main_file_path = llm_response["context"][0].metadata["source"]

            # 補足メッセージの表示
            main_message = "入力内容に関する情報は、以下のファイルに含まれている可能性があります。"
            st.markdown(main_message)
            
            # メインドキュメントの表示情報を準備
            main_file_info = main_file_path
            main_page_number = None
            
            # ページ番号が取得できた場合、ファイル情報に追加
            if "page" in llm_response["context"][0].metadata:
                main_page_number = llm_response["context"][0].metadata["page"]
                main_file_info = f"{main_file_path}（Page #{main_page_number}）"
            
            # メインドキュメントのアイコンを取得
            icon = utils.get_source_icon(main_file_path)
            st.success(main_file_info, icon=icon)

            # ==========================================
            # ユーザー入力値と関連性が高いサブドキュメントのありかを表示
            # ==========================================
            # メインドキュメント以外で、関連性が高いサブドキュメントを格納する用のリスト
            sub_choices = []
            # 重複チェック用のリスト
            duplicate_check_list = []

            # ドキュメントが2件以上検索できた場合（サブドキュメントが存在する場合）のみ、サブドキュメントのありかを一覧表示
            if len(llm_response["context"]) > 1:
                for document in llm_response["context"][1:]:
                    # ドキュメントのファイルパスを取得
                    sub_file_path = document.metadata["source"]

                    # メインドキュメントのファイルパスと重複している場合、処理をスキップ（表示しない）
                    if sub_file_path == main_file_path:
                        continue
                    
                    # 同じファイル内の異なる箇所を参照した場合、2件目以降のファイルパスに重複が発生する可能性があるため、重複を除去
                    if sub_file_path in duplicate_check_list:
                        continue

                    # 重複チェック用のリストにファイルパスを順次追加
                    duplicate_check_list.append(sub_file_path)
                    
                    # サブドキュメントの表示情報を準備
                    page_number = None
                    # ページ番号が取得できた場合のみ追加
                    if "page" in document.metadata:
                        page_number = document.metadata["page"]
                    
                    # 後ほど一覧表示するため、サブドキュメントに関する情報を順次リストに追加
                    sub_choices.append({
                        "source": sub_file_path,
                        "page_number": page_number
                    })
            
            # サブドキュメントが存在する場合のみの処理
            sub_message = None
            if sub_choices:
                st.markdown("##### 関連資料")
                sub_message = "その他、参考になりそうな資料はこちらです。"
                st.markdown(sub_message)

                # サブドキュメントに対してのループ処理
                for sub_choice in sub_choices:
                    sub_info = sub_choice['source']
                    if sub_choice['page_number']:
                        sub_info += f"（Page #{sub_choice['page_number']}）"
                    
                    # 参照元のアイコンを取得して表示
                    icon = utils.get_source_icon(sub_choice['source'])
                    st.info(sub_info, icon=icon)

            # 表示用の会話ログに格納するためのデータを用意
            content = {}
            content["mode"] = ct.ANSWER_MODE_1
            content["main_message"] = main_message
            content["main_file_path"] = main_file_path
            
            # メインドキュメントのページ番号は、取得できた場合にのみ追加
            if main_page_number:
                content["main_page_number"] = main_page_number
            
            # サブドキュメントの情報は、取得できた場合にのみ追加
            if sub_choices:
                content["sub_message"] = sub_message
                content["sub_choices"] = sub_choices
            
            return content
            
        except Exception as e:
            logger.error(f"検索モード応答表示エラー: {e}")
            st.markdown(ct.NO_DOC_MATCH_MESSAGE)
            return {
                "mode": ct.ANSWER_MODE_1,
                "answer": ct.NO_DOC_MATCH_MESSAGE,
                "no_file_path_flg": True
            }
    
    # LLMからのレスポンスに、ユーザー入力値と関連性の高いドキュメント情報が入って「いない」場合
    else:
        # 関連ドキュメントが取得できなかった場合のメッセージ表示
        st.markdown(ct.NO_DOC_MATCH_MESSAGE)

        # 表示用の会話ログに格納するためのデータを用意
        return {
            "mode": ct.ANSWER_MODE_1,
            "answer": ct.NO_DOC_MATCH_MESSAGE,
            "no_file_path_flg": True
        }


def display_contact_llm_response(llm_response):
    """
    「社内問い合わせ」モードにおけるLLMレスポンスを表示

    Args:
        llm_response: LLMからの回答

    Returns:
        LLMからの回答を画面表示用に整形した辞書データ
    """
    logger = logging.getLogger(ct.LOGGER_NAME)
    
    try:
        # LLMからの回答を表示
        st.markdown(llm_response["answer"])
        
        # 表示用の会話ログに格納するためのデータを用意
        content = {
            "mode": ct.ANSWER_MODE_2,
            "answer": llm_response["answer"]
        }
        
        # 参照元の文書情報がある場合は追加
        if "context" in llm_response and llm_response["context"]:
            file_info_list = []
            duplicate_check_list = []
            
            # 参照元のドキュメント情報をリストに追加
            for document in llm_response["context"]:
                # ドキュメントのファイルパスを取得
                file_path = document.metadata["source"]
                
                # 重複を除去
                if file_path in duplicate_check_list:
                    continue
                    
                # 重複チェック用のリストにファイルパスを追加
                duplicate_check_list.append(file_path)
                
                # ファイル情報を構築
                file_info = file_path
                if "page" in document.metadata:
                    file_info = f"{file_path}（Page #{document.metadata['page']}）"
                    
                file_info_list.append(file_info)
            
            # 参照元情報がある場合のみ、表示用の会話ログに追加
            if file_info_list:
                message = "情報源"
                content["message"] = message
                content["file_info_list"] = file_info_list
                
                # 区切り線
                st.divider()
                
                # 「情報源」の見出し表示
                st.markdown(f"##### {message}")
                
                # 参照元ドキュメントの一覧表示
                for file_info in file_info_list:
                    icon = utils.get_source_icon(file_info)
                    st.info(file_info, icon=icon)
                    
        return content
        
    except Exception as e:
        logger.error(f"問い合わせモード応答表示エラー: {e}")
        error_message = "回答の表示中にエラーが発生しました。もう一度お試しください。"
        st.error(error_message)
        return {
            "mode": ct.ANSWER_MODE_2,
            "answer": error_message
        }