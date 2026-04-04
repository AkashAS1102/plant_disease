# 🎓 EduAI - Study Smarter

EduAI is a modern, intelligent web application designed to help students and professionals study smarter by leveraging advanced AI models. With EduAI, you can upload your study materials, generate comprehensive summaries, test your knowledge with interactive quizzes, and engage in active recall using spaced-repetition flashcards. 

Powered by **React**, **Vite**, and the ultra-fast **Groq API (Llama 3.3)**, EduAI delivers instant, high-quality educational assistance.

## ✨ Features

- **📤 Upload & Summarize**: Upload PDF documents and get an instant, structured AI-generated summary. Break down long lectures or dense reading into easy-to-digest bullet points.
- **🎯 Dynamic Quizzes**: Automatically generate multiple-choice quizzes based on your uploaded documents to test your reading comprehension.
- **🃏 Smart Flashcards**: Create flashcards for active recall and spaced repetition (based on the SM-2 algorithm).
- **💬 Ask AI**: Chat directly with your documents. Ask specific questions, clarify complex topics, or ask for explanations just like you would with a tutor.
- **🌓 Dark Mode**: Built-in toggle for comfortable late-night studying sessions.

## 🚀 Tech Stack

- **Frontend Framework**: [React 18](https://reactjs.org/) + [Vite](https://vitejs.dev/)
- **Styling**: [Tailwind CSS](https://tailwindcss.com/) & custom CSS
- **AI / LLM Integration**: [Groq API](https://groq.com/) (using Llama 3.3 for lightning-fast inference)
- **Database / Backend**: [Supabase](https://supabase.com/)
- **PDF Processing**: PDF.js (`pdfjs-dist`)
- **Icons**: Lucide React

## ⚙️ Prerequisites

Before you begin, ensure you have the following installed:
- [Node.js](https://nodejs.org/) (v18 or higher recommended)
- npm or yarn

You will also need to create accounts and get API keys from:
- **Groq Console** (for the AI integration)
- **Supabase** (for the backend data storage)

## 🛠️ Installation & Setup

1. **Clone the repository** (if applicable):
   ```bash
   git clone https://github.com/your-username/eduai.git
   cd eduai
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Configure Environment Variables**:
   Create a `.env` file in the root directory based on the provided sample, and fill in your keys:
   ```env
   # .env
   VITE_SUPABASE_URL=https://your-project.supabase.co
   VITE_SUPABASE_ANON_KEY=your-supabase-anon-key
   
   VITE_GROQ_API_KEY=your-groq-api-key
   ```

4. **Start the Development Server**:
   ```bash
   npm run dev
   ```
   The application will be running at `http://localhost:5173`.

## 📦 Building for Production

To create a production build, run:
```bash
npm run build
```
This will compile the application and output the optimized files in the `dist` folder. You can test the production build using:
```bash
npm run preview
```

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/your-username/eduai/issues).

## 📄 License

This project is licensed under the MIT License.
