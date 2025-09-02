import React, { useState } from 'react';
import { TextField, Button, Typography, Box } from '@mui/material';

export default function InputForm({ onSubmit }) {
  const [youtubeLink, setYoutubeLink] = useState('');
  const [pdfFile, setPdfFile] = useState(null);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!youtubeLink && !pdfFile) {
      alert('Please provide a YouTube link or upload a PDF.');
      return;
    }
    onSubmit({ youtubeLink, pdfFile });
  };

  return (
    <Box component="form" onSubmit={handleSubmit} sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      <Typography variant="h4" align="center">Quiz Generator</Typography>
      <TextField label="YouTube Link" value={youtubeLink} onChange={(e) => setYoutubeLink(e.target.value)} fullWidth />
      <Typography align="center">or</Typography>
      <Button variant="contained" component="label">
        Upload PDF
        <input type="file" hidden accept="application/pdf" onChange={(e) => setPdfFile(e.target.files[0])} />
      </Button>
      {pdfFile && <Typography>Selected: {pdfFile.name}</Typography>}
      <Button type="submit" variant="contained" color="primary">Generate Quiz</Button>
    </Box>
  );
}
