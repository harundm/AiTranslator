import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

app = FastAPI(title="Gemini Translation API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Helper function to load API key
def get_api_key():
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        return api_key
    
    # Try local .env file in the same directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(current_dir, ".env")
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            content = f.read().strip()
            if content and "=" not in content:
                return content
            # If it's formatted as KEY=VALUE
            for line in content.splitlines():
                if line.startswith("GEMINI_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
                
    # Try local apikey file
    apikey_path = os.path.join(current_dir, "apikey")
    if os.path.exists(apikey_path):
        with open(apikey_path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
            # From earlier, apikey has the key on lines 1 and 3
            if lines:
                return lines[0]
                
    return None

api_key = get_api_key()
if not api_key:
    # Use the hardcoded key from example files as fallback
    api_key = "AQ.Ab8RN6JFqgtLjAgHr-flgtlWQ0yEYjYCdEswOicF4Fw90LRjRw"

# Initialize LangChain model
try:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=api_key)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a highly accurate, professional translator. Translate the given text from {source_language} to {target_language}. Maintain the tone, style, and formatting of the original text. Only return the translated text without any explanation, markdown formatting (unless present in the original), or extra comments."
            ),
            (
                "human",
                "{text}"
            ),
        ]
    )
    chain = prompt | llm
except Exception as e:
    print(f"Error initializing Google Generative AI: {e}")
    chain = None

class TranslationRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to translate")
    source_language: str = Field("English", description="Source language of the text")
    target_language: str = Field("German", description="Target language for translation")

class TranslationResponse(BaseModel):
    original_text: str
    translated_text: str
    source_language: str
    target_language: str

@app.post("/translate", response_model=TranslationResponse)
async def translate(request: TranslationRequest):
    if not chain:
        raise HTTPException(status_code=500, detail="Translation engine is not initialized. Please verify your API key.")
    
    try:
        # Run translation chain
        ai_msg = chain.invoke(
            {
                "source_language": request.source_language,
                "target_language": request.target_language,
                "text": request.text
            }
        )
        translated_text = ai_msg.content.strip()
        return TranslationResponse(
            original_text=request.text,
            translated_text=translated_text,
            source_language=request.source_language,
            target_language=request.target_language
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def get_root():
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="description" content="AI Translation Studio - Translate text accurately with Gemini-2.5-Flash">
    <title>AI Translation Studio</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #8a2be2;
            --primary-glow: rgba(138, 43, 226, 0.4);
            --bg-color: #0b0813;
            --card-bg: rgba(20, 16, 35, 0.65);
            --card-border: rgba(255, 255, 255, 0.08);
            --text-primary: #ffffff;
            --text-secondary: #a0aec0;
            --input-bg: rgba(255, 255, 255, 0.04);
            --input-border: rgba(255, 255, 255, 0.1);
            --success: #10b981;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Outfit', sans-serif;
            -webkit-font-smoothing: antialiased;
        }

        body {
            background-color: var(--bg-color);
            background-image: 
                radial-gradient(circle at 10% 20%, rgba(138, 43, 226, 0.15) 0%, transparent 40%),
                radial-gradient(circle at 90% 80%, rgba(0, 194, 255, 0.1) 0%, transparent 45%);
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 2rem;
            overflow-x: hidden;
        }

        .container {
            width: 100%;
            max-width: 900px;
            backdrop-filter: blur(20px);
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 24px;
            padding: 3rem;
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.4);
            animation: fadeIn 0.8s ease-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        header {
            text-align: center;
            margin-bottom: 2.5rem;
        }

        .logo-badge {
            display: inline-block;
            padding: 8px 16px;
            background: rgba(138, 43, 226, 0.15);
            border: 1px solid rgba(138, 43, 226, 0.3);
            border-radius: 50px;
            color: #d8b4fe;
            font-size: 0.85rem;
            font-weight: 600;
            letter-spacing: 1px;
            text-transform: uppercase;
            margin-bottom: 1rem;
            box-shadow: 0 0 15px rgba(138, 43, 226, 0.1);
        }

        h1 {
            font-size: 2.75rem;
            font-weight: 700;
            background: linear-gradient(135deg, #ffffff 40%, #c084fc 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
            letter-spacing: -0.5px;
        }

        header p {
            color: var(--text-secondary);
            font-size: 1.1rem;
            font-weight: 300;
        }

        .workspace {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
            margin-bottom: 2rem;
        }

        @media (max-width: 768px) {
            .workspace {
                grid-template-columns: 1fr;
            }
            .container {
                padding: 1.5rem;
            }
            h1 {
                font-size: 2.2rem;
            }
        }

        .panel {
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }

        .panel-label {
            font-size: 0.9rem;
            font-weight: 600;
            color: #d8b4fe;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        select {
            width: 100%;
            padding: 12px 16px;
            background: var(--input-bg);
            border: 1px solid var(--input-border);
            border-radius: 12px;
            color: var(--text-primary);
            font-size: 1rem;
            font-weight: 500;
            outline: none;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        select:focus {
            border-color: var(--primary);
            box-shadow: 0 0 10px var(--primary-glow);
        }

        textarea {
            width: 100%;
            height: 220px;
            padding: 16px;
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--input-border);
            border-radius: 16px;
            color: var(--text-primary);
            font-size: 1.05rem;
            line-height: 1.6;
            resize: none;
            outline: none;
            transition: all 0.3s ease;
        }

        textarea:focus {
            background: rgba(255, 255, 255, 0.04);
            border-color: var(--primary);
            box-shadow: 0 0 15px var(--primary-glow);
        }

        textarea::placeholder {
            color: #636b77;
        }

        .output-box {
            position: relative;
            background: rgba(0, 0, 0, 0.2);
        }

        .output-content {
            height: 220px;
            padding: 16px;
            border: 1px solid var(--input-border);
            border-radius: 16px;
            color: var(--text-primary);
            font-size: 1.05rem;
            line-height: 1.6;
            overflow-y: auto;
            white-space: pre-wrap;
            transition: all 0.3s ease;
        }

        .output-content.loading {
            opacity: 0.6;
        }

        .copy-btn {
            position: absolute;
            bottom: 12px;
            right: 12px;
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid var(--input-border);
            border-radius: 8px;
            color: var(--text-secondary);
            padding: 6px 12px;
            font-size: 0.8rem;
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .copy-btn:hover {
            background: rgba(255, 255, 255, 0.15);
            color: var(--text-primary);
        }

        .copy-btn.copied {
            background: var(--success);
            color: white;
            border-color: var(--success);
        }

        .action-area {
            display: flex;
            justify-content: center;
            margin-top: 1rem;
        }

        .translate-btn {
            background: linear-gradient(135deg, #a855f7 0%, #7c3aed 100%);
            color: white;
            border: none;
            padding: 14px 40px;
            font-size: 1.1rem;
            font-weight: 600;
            border-radius: 14px;
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            box-shadow: 0 10px 20px rgba(124, 58, 237, 0.3);
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .translate-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 15px 25px rgba(124, 58, 237, 0.45);
        }

        .translate-btn:active {
            transform: translateY(1px);
        }

        .translate-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }

        /* Spinner style */
        .spinner {
            width: 18px;
            height: 18px;
            border: 2px solid rgba(255, 255, 255, 0.3);
            border-top: 2px solid white;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
            display: none;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .character-count {
            font-size: 0.8rem;
            color: #636b77;
            text-align: right;
            margin-top: -4px;
        }

        /* Custom Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
        }
        ::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.02);
            border-radius: 8px;
        }
        ::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 8px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: rgba(255, 255, 255, 0.2);
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="logo-badge">Powered by Gemini 2.5 Flash</div>
            <h1 id="title-main">AI Translation Studio</h1>
            <p>Sleek, real-time text translation using Google Generative AI</p>
        </header>

        <div class="workspace">
            <div class="panel">
                <div class="panel-label">
                    <span>Source Language</span>
                </div>
                <select id="source-lang">
                    <option value="English" selected>English</option>
                    <option value="Spanish">Spanish</option>
                    <option value="French">French</option>
                    <option value="German">German</option>
                    <option value="Chinese">Chinese</option>
                    <option value="Japanese">Japanese</option>
                    <option value="Korean">Korean</option>
                    <option value="Italian">Italian</option>
                    <option value="Portuguese">Portuguese</option>
                    <option value="Russian">Russian</option>
                    <option value="Bahasa Malaysia">Bahasa Malaysia</option>
                </select>
                <textarea id="source-text" placeholder="Type or paste text to translate here..." maxlength="5000"></textarea>
                <div class="character-count" id="char-counter">0 / 5000</div>
            </div>

            <div class="panel">
                <div class="panel-label">
                    <span>Target Language</span>
                </div>
                <select id="target-lang">
                    <option value="English">English</option>
                    <option value="Spanish">Spanish</option>
                    <option value="French">French</option>
                    <option value="German" selected>German</option>
                    <option value="Chinese">Chinese</option>
                    <option value="Japanese">Japanese</option>
                    <option value="Korean">Korean</option>
                    <option value="Italian">Italian</option>
                    <option value="Portuguese">Portuguese</option>
                    <option value="Russian">Russian</option>
                    <option value="Bahasa Malaysia">Bahasa Malaysia</option>
                </select>
                <div class="output-box">
                    <div class="output-content" id="output-text" placeholder="Translation will appear here..."></div>
                    <button class="copy-btn" id="copy-button" aria-label="Copy to Clipboard">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                            <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                            <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                        </svg>
                        <span>Copy</span>
                    </button>
                </div>
            </div>
        </div>

        <div class="action-area">
            <button class="translate-btn" id="translate-button">
                <div class="spinner" id="btn-spinner"></div>
                <span id="btn-text">Translate</span>
            </button>
        </div>
    </div>

    <script>
        const sourceText = document.getElementById('source-text');
        const outputText = document.getElementById('output-text');
        const translateBtn = document.getElementById('translate-button');
        const btnSpinner = document.getElementById('btn-spinner');
        const btnText = document.getElementById('btn-text');
        const sourceLang = document.getElementById('source-lang');
        const targetLang = document.getElementById('target-lang');
        const charCounter = document.getElementById('char-counter');
        const copyBtn = document.getElementById('copy-button');

        // Character limit counter
        sourceText.addEventListener('input', () => {
            charCounter.textContent = `${sourceText.value.length} / 5000`;
        });

        // Translate trigger
        translateBtn.addEventListener('click', async () => {
            const text = sourceText.value.trim();
            if (!text) return;

            // Loading state
            translateBtn.disabled = true;
            btnSpinner.style.display = 'block';
            btnText.textContent = 'Translating...';
            outputText.classList.add('loading');
            outputText.textContent = 'Analyzing and translating text...';

            try {
                const response = await fetch('/translate', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        text: text,
                        source_language: sourceLang.value,
                        target_language: targetLang.value
                    })
                });

                if (!response.ok) {
                    const errData = await response.json();
                    throw new Error(errData.detail || 'Translation error occurred');
                }

                const data = await response.json();
                outputText.textContent = data.translated_text;
            } catch (err) {
                outputText.innerHTML = `<span style="color: #ef4444;">Error: ${err.message}</span>`;
            } finally {
                // Clear loading state
                translateBtn.disabled = false;
                btnSpinner.style.display = 'none';
                btnText.textContent = 'Translate';
                outputText.classList.remove('loading');
            }
        });

        // Copy button trigger
        copyBtn.addEventListener('click', () => {
            const text = outputText.textContent;
            if (!text || text.startsWith('Translation will') || text.startsWith('Error:')) return;

            navigator.clipboard.writeText(text).then(() => {
                const span = copyBtn.querySelector('span');
                span.textContent = 'Copied!';
                copyBtn.classList.add('copied');

                setTimeout(() => {
                    span.textContent = 'Copy';
                    copyBtn.classList.remove('copied');
                }, 2000);
            });
        });
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
