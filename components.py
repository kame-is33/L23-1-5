"""
このファイルは、画面表示に特化した関数定義のファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
import streamlit as st
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


def display_select_mode():
    """
    回答モードのラジオボタンを表示
    """
    # サイドバーにラジオボタンを表示
    st.sidebar.markdown("### 利用目的")
    st.session_state.mode = st.sidebar.radio(
        label="",
        options=[ct.ANSWER_MODE_1, ct.ANSWER_MODE_2],
        label_visibility="collapsed"
    )


def display_sidebar():
    """
    サイドバーの表示 (新設)
    """
    # 「社内文書検索」の機能説明
    st.sidebar.markdown(ct.SIDEBAR_SEARCH_TITLE)
    # 「st.info()」を使うと青枠で表示される
    st.sidebar.markdown(ct.SIDEBAR_SEARCH_DESCRIPTION)
    # 「st.code()」を使うとコードブロックの装飾で表示される
    # 「wrap_lines=True」で折り返し設定、「language=None」で非装飾とする
    st.sidebar.markdown(ct.EXAMPLE_TITLE)
    st.sidebar.code(ct.SIDEBAR_SEARCH_EXAMPLE, wrap_lines=True, language=None)

    # 「社内問い合わせ」の機能説明
    st.sidebar.markdown(ct.SIDEBAR_INQUIRY_TITLE)
    st.sidebar.markdown(ct.SIDEBAR_INQUIRY_DESCRIPTION)
    st.sidebar.code(ct.SIDEBAR_INQUIRY_EXAMPLE, wrap_lines=True, language=None)

    # 社員情報に関する説明
    st.sidebar.markdown(ct.SIDEBAR_EMPLOYEE_TITLE)
    st.sidebar.markdown(ct.SIDEBAR_EMPLOYEE_DESCRIPTION)
    st.sidebar.code(ct.SIDEBAR_EMPLOYEE_EXAMPLE, wrap_lines=True, language=None)

    # 区切り線
    st.sidebar.divider()
        
    # 一番下に開発者モードのトグル
    st.sidebar.divider()
    
    # セッション状態に開発者モードのフラグが存在しない場合は初期化
    if "developer_mode" not in st.session_state:
        st.session_state.developer_mode = False
    
    # 開発者モードのトグル
    st.session_state.developer_mode = st.sidebar.toggle(
        ct.SIDEBAR_DEVELOPER_MODE,
        value=st.session_state.developer_mode,
        key="developer_mode_sidebar_toggle"
    )


def display_initial_ai_message():
    """
    AIメッセージの初期表示
    """
    with st.chat_message("assistant"):
        # 「st.success()」とすると緑枠で表示される
        st.markdown("こんにちは。私は社内文書の情報をもとに回答する生成AIチャットボットです。上記で利用目的を選択し、画面下部のチャット欄からメッセージを送信してください。")


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

                        # 参照元のありかに応じて、適したアイコンを取得
                        icon = utils.get_source_icon(message['content']['main_file_path'])
                        # 参照元ドキュメントのページ番号が取得できた場合にのみ、ページ番号を表示
                        if "main_page_number" in message["content"]:
                            st.success(f"{message['content']['main_file_path']}", icon=icon)
                        else:
                            st.success(f"{message['content']['main_file_path']}", icon=icon)
                        
                        # ==========================================
                        # ユーザー入力値と関連性が高いサブドキュメントのありかを表示
                        # ==========================================
                        if "sub_message" in message["content"]:
                            # 補足メッセージの表示
                            st.markdown(message["content"]["sub_message"])

                            # サブドキュメントのありかを一覧表示
                            for sub_choice in message["content"]["sub_choices"]:
                                # 参照元のありかに応じて、適したアイコンを取得
                                icon = utils.get_source_icon(sub_choice['source'])
                                # 参照元ドキュメントのページ番号が取得できた場合にのみ、ページ番号を表示
                                if "page" in sub_choice:
                                    st.info(f"{sub_choice['source']}", icon=icon)
                                else:
                                    st.info(f"{sub_choice['source']}", icon=icon)
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
    # 開発者モードがオンの場合、デバッグ情報を表示
    if st.session_state.developer_mode:
        with st.expander(ct.DEBUG_EXPANDER_TITLE):
            st.markdown(ct.DEBUG_LLM_RESPONSE_TITLE)
            st.json(llm_response)
    
    # LLMからのレスポンスに参照元情報が入っており、かつ「該当資料なし」が回答として返された場合
    if llm_response["context"] and llm_response["answer"] != ct.NO_DOC_MATCH_ANSWER:
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
        # メインドキュメント以外で、関連性が高いサブドキュメントを格納する用のリストを用意
        sub_choices = []
        # 重複チェック用のリストを用意
        duplicate_check_list = []

        # ドキュメントが2件以上検索できた場合（サブドキュメントが存在する場合）のみ、サブドキュメントのありかを一覧表示
        # 「source_documents」内のリストの2番目以降をスライスで参照（2番目以降がなければfor文内の処理は実行されない）
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
            sub_file_info = sub_file_path
            # ページ番号が取得できた場合、ファイル情報に追加
            if "page" in document.metadata:
                page_number = document.metadata["page"]
                sub_file_info = f"{sub_file_path}（Page #{page_number}）"
            
            # 参照元のありかに応じて、適したアイコンを取得
            icon = utils.get_source_icon(sub_file_path)
            st.info(sub_file_info, icon=icon)
            
            # 後ほど一覧表示するため、サブドキュメントに関する情報を順次リストに追加
            sub_choices.append({
                "source": sub_file_path,
                "page_number": page_number if "page" in document.metadata else None
            })
        
        # サブドキュメントが存在する場合のみの処理
        if sub_choices:
            # 補足メッセージの表示
            sub_message = "その他、ファイルありかの候補を提示します。"
            st.markdown(sub_message)

        # 表示用の会話ログに格納するためのデータを用意
        content = {}
        content["mode"] = ct.ANSWER_MODE_1
        content["main_message"] = main_message
        content["main_file_path"] = main_file_path
        
        # メインドキュメントのページ番号は、取得できた場合にのみ追加
        if "page" in llm_response["context"][0].metadata:
            content["main_page_number"] = main_page_number
        
        # サブドキュメントの情報は、取得できた場合にのみ追加
        if sub_choices:
            content["sub_message"] = sub_message
            content["sub_choices"] = sub_choices
    
    # LLMからのレスポンスに、ユーザー入力値と関連性の高いドキュメント情報が入って「いない」場合
    else:
        # 関連ドキュメントが取得できなかった場合のメッセージ表示
        st.markdown(ct.NO_DOC_MATCH_MESSAGE)

        # 表示用の会話ログに格納するためのデータを用意
        content = {}
        content["mode"] = ct.ANSWER_MODE_1
        content["answer"] = ct.NO_DOC_MATCH_MESSAGE
        content["no_file_path_flg"] = True
    
    return content


def display_contact_llm_response(llm_response):
    """
    「社内問い合わせ」モードにおけるLLMレスポンスを表示

    Args:
        llm_response: LLMからの回答

    Returns:
        LLMからの回答を画面表示用に整形した辞書データ
    """
    # 開発者モードがオンの場合、デバッグ情報を表示
    if st.session_state.developer_mode:
        with st.expander(ct.DEBUG_EXPANDER_TITLE):
            st.markdown(ct.DEBUG_LLM_RESPONSE_TITLE)
            st.json(llm_response)
    
    # LLMからの回答を表示
    st.markdown(llm_response["answer"])
    
    # 表示用の会話ログに格納するためのデータを用意
    content = {}
    content["mode"] = ct.ANSWER_MODE_2
    content["answer"] = llm_response["answer"]
    
    # 参照元の文書情報がある場合は追加
    if llm_response["context"] and llm_response["answer"] != ct.INQUIRY_NO_MATCH_ANSWER:
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
            
            # ファイル情報をリストに追加
            if "page" in document.metadata:
                file_info = f"{file_path}（Page #{document.metadata['page']}）"
            else:
                file_info = file_path
                
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