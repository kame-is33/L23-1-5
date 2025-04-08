"""
このファイルは、固定の文字列や数値などのデータを変数として一括管理するファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader, TextLoader
from langchain_community.document_loaders.csv_loader import CSVLoader


############################################################
# 共通変数の定義
############################################################

# ==========================================
# 画面表示系
# ==========================================
APP_NAME = "社内情報特化型生成AI検索アプリ"
ANSWER_MODE_1 = "社内文書検索"  # ANSWER_MODE_1 = "社内文書検索" → SYSTEM_PROMPT_DOC_SEARCH に対応
ANSWER_MODE_2 = "社内問い合わせ"  # ANSWER_MODE_2 = "社内問い合わせ" → SYSTEM_PROMPT_INQUIRY に対応
CHAT_INPUT_HELPER_TEXT = "こちらからメッセージを送信してください。"
DOC_SOURCE_ICON = ":material/description: "
LINK_SOURCE_ICON = ":material/link: "
WARNING_ICON = ":material/warning:"
ERROR_ICON = ":material/error:"
SPINNER_TEXT = "回答生成中..."


# ==========================================
# サイドバー表示系
# ==========================================
EXAMPLE_TITLE = "**【入力例】**"
SIDEBAR_SEARCH_TITLE = "**『社内文書検索』を選択した場合**"
SIDEBAR_SEARCH_DESCRIPTION = "入力内容と関連性が高い社内文書のありかを検索できます。"
SIDEBAR_SEARCH_EXAMPLE = "社員の育成方針に関するMTGの議事録"

SIDEBAR_INQUIRY_TITLE = "**『社内問い合わせ』を選択した場合**"
SIDEBAR_INQUIRY_DESCRIPTION = "質問・要望に対して、社内文書の情報をもとに回答を得られます。"
SIDEBAR_INQUIRY_EXAMPLE = "人事部に所属している従業員情報を一覧化して"

SIDEBAR_EMPLOYEE_TITLE = "**【社員情報を含む質問】**"
SIDEBAR_EMPLOYEE_DESCRIPTION = "人事・従業員・部署に関する質問をすると、社員名簿のデータを参照して回答します。"
SIDEBAR_EMPLOYEE_EXAMPLE = "人事部に所属する全従業員のスキルセットを一覧にしてください"

SIDEBAR_DEVELOPER_MODE = "開発者モード"
DEBUG_EXPANDER_TITLE = "DEBUG情報"
DEBUG_LLM_RESPONSE_TITLE = "### LLMレスポンス（生データ）"

# ==========================================
# ログ出力系
# ==========================================
LOG_DIR_PATH = "./logs"
LOGGER_NAME = "ApplicationLog"
LOG_FILE = "application.log"
APP_BOOT_MESSAGE = "アプリが起動されました。"


# ==========================================
# LLM設定系
# ==========================================
MODEL = "gpt-4o-mini"
TEMPERATURE = 0.5


# ==========================================
# RAG参照用のデータソース系
# ==========================================
RAG_TOP_FOLDER_PATH = "./data"
SUPPORTED_EXTENSIONS = {
    ".pdf": lambda path: PyMuPDFLoader(path),
    ".docx": lambda path: Docx2txtLoader(path),
    ".csv": lambda path: CSVLoader(path, encoding="utf-8"),
    ".txt": lambda path: TextLoader(path, encoding="utf-8")
}
WEB_URL_LOAD_TARGETS = [
    "https://generative-ai.web-camp.io/"
]

# ==========================================
# RAG設定系　追加
# ==========================================
RETRIEVER_DOCUMENT_COUNT_DEFAULT = 5   # 通常の検索結果として取得するドキュメント数
RETRIEVER_DOCUMENT_COUNT_EMPLOYEE = 15  # 社員情報クエリ時の検索結果として取得するドキュメント数
CHUNK_SIZE = 1000                 # チャンク分割サイズ（500から1000に増加）
CHUNK_OVERLAP = 100               # チャンク分割時のオーバーラップサイズ（50から100に増加）


# ==========================================
# プロンプトテンプレート
# ==========================================
SYSTEM_PROMPT_CREATE_INDEPENDENT_TEXT = "会話履歴と最新の入力をもとに、会話履歴なしでも理解できる独立した入力テキストを生成してください。"

SYSTEM_PROMPT_DOC_SEARCH = """
    あなたは社内の文書検索アシスタントです。
    以下の条件に基づき、ユーザー入力に対して回答してください。

    【条件】
    1. ユーザー入力内容と以下の文脈との間に関連性がある場合、空文字「""」を返してください。
    2. ユーザー入力内容と以下の文脈との関連性が明らかに低い場合、「該当資料なし」と回答してください。

    【ユーザーの質問】
    {user_question}

    【文脈】
    {context}
"""

SYSTEM_PROMPT_INQUIRY = """
    あなたは社内情報特化型のアシスタントです。
    以下の条件に基づき、ユーザー入力に対して回答してください。

    【条件】
    1. ユーザー入力内容と以下の文脈との間に関連性がある場合のみ、以下の文脈に基づいて回答してください。
    2. ユーザー入力内容と以下の文脈との関連性が明らかに低い場合、「回答に必要な情報が見つかりませんでした。」と回答してください。
    3. 憶測で回答せず、あくまで以下の文脈を元に回答してください。
    4. できる限り詳細に、マークダウン記法を使って回答してください。
    5. マークダウン記法で回答する際にhタグの見出しを使う場合、最も大きい見出しをh3としてください。
    6. 複雑な質問の場合、各項目についてそれぞれ詳細に回答してください。
    7. 必要と判断した場合は、以下の文脈に基づかずとも、一般的な情報を回答してください。

    【ユーザーの質問】
    {user_question}

    【文脈】
    {context}
"""

# 社員情報に特化したプロンプト（新規追加）
SYSTEM_PROMPT_EMPLOYEE = """
    あなたは社内の人事情報に特化したアシスタントです。
    以下の社員名簿データを基に質問に回答してください。

    【条件】
    1. 以下の社員名簿データのみを使用して回答してください。
    2. データは表形式です。適切に整形して回答してください。
    3. 複数行のデータがある場合は、すべての行を考慮してください。
    4. 部署や役職でフィルタリングする場合は、該当するすべての社員の情報を含めてください。
    5. スキルセットや特性について質問された場合は、関連するすべての情報を表形式で整理して回答してください。
    6. マークダウン記法を使って見やすく整形してください。
    7. 回答には、どのデータを参照したかを明示してください。

    【ユーザーの質問】
    {user_question}

    【社員名簿データ】
    {employee_data}
"""


# ==========================================
# LLMレスポンスの一致判定用
# ==========================================
INQUIRY_NO_MATCH_ANSWER = "回答に必要な情報が見つかりませんでした。"
NO_DOC_MATCH_ANSWER = "該当資料なし"


# ==========================================
# エラー・警告メッセージ
# ==========================================
COMMON_ERROR_MESSAGE = "このエラーが繰り返し発生する場合は、管理者にお問い合わせください。"
INITIALIZE_ERROR_MESSAGE = "初期化処理に失敗しました。"
NO_DOC_MATCH_MESSAGE = """
    入力内容と関連する社内文書が見つかりませんでした。\n
    入力内容を変更してください。
"""
CONVERSATION_LOG_ERROR_MESSAGE = "過去の会話履歴の表示に失敗しました。"
GET_LLM_RESPONSE_ERROR_MESSAGE = "回答生成に失敗しました。"
DISP_ANSWER_ERROR_MESSAGE = "回答表示に失敗しました。"
EMPLOYEE_DATA_ERROR_MESSAGE = "社員情報の処理中にエラーが発生しました。"


# ==========================================
# ファイル更新検知系　新設
# ==========================================
FILE_UPDATE_MESSAGE = "データソースの更新を検知しました。最新の情報を反映します。"
FILE_UPDATE_ICON = "ℹ️"

# ==========================================
# 社員情報処理用キーワード（新規追加）
# ==========================================
EMPLOYEE_KEYWORDS = ["人事", "従業員", "社員", "部署", "スキル", "名簿", "所属"]