import express from 'express';
import multer from 'multer';
import pdfParse from 'pdf-parse';
import { YoutubeTranscript } from 'youtube-transcript';
import cors from 'cors';

const app = express();
app.use(cors());
const upload = multer();

function extractVideoId(url) {
  const reg = /(?:v=|\.be\/)([\w-]{11})/;
  const match = url.match(reg);
  return match ? match[1] : null;
}

function generateQuestions(text) {
  const sentences = text
    .replace(/\s+/g, ' ')
    .split(/[.!?]/)
    .map((s) => s.trim())
    .filter(Boolean);
  const questions = [];
  for (let i = 0; i < Math.min(10, sentences.length); i++) {
    const words = sentences[i].split(' ');
    if (words.length < 2) continue;
    const answer = words.pop();
    questions.push({
      question: words.join(' ') + ' ...?',
      answer
    });
  }
  return questions;
}

app.post('/api/quiz', upload.single('pdf'), async (req, res) => {
  try {
    let text = '';
    if (req.body.youtubeLink) {
      const id = extractVideoId(req.body.youtubeLink);
      if (!id) return res.status(400).json({ error: 'Invalid YouTube link' });
      const transcript = await YoutubeTranscript.fetchTranscript(id);
      text = transcript.map((t) => t.text).join(' ');
    } else if (req.file) {
      const data = await pdfParse(req.file.buffer);
      text = data.text;
    } else {
      return res.status(400).json({ error: 'No input provided' });
    }

    const questions = generateQuestions(text);
    if (questions.length === 0) {
      return res.status(400).json({ error: 'Could not generate questions' });
    }
    res.json({ questions });
  } catch (err) {
    res.status(500).json({ error: 'Processing failed' });
  }
});

const PORT = 3000;
app.listen(PORT, () => console.log(`Server running on ${PORT}`));
