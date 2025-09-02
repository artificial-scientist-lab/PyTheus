import React, { useState } from 'react';
import { Box, Button, TextField, Typography } from '@mui/material';

export default function Quiz({ questions, onReset }) {
  const [answers, setAnswers] = useState(Array(questions.length).fill(''));
  const [results, setResults] = useState(null);

  const handleChange = (index, value) => {
    const arr = [...answers];
    arr[index] = value;
    setAnswers(arr);
  };

  const handleCheck = () => {
    const res = questions.map((q, i) => q.answer.toLowerCase() === answers[i].trim().toLowerCase());
    setResults(res);
  };

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      <Typography variant="h5" align="center">Your Quiz</Typography>
      {questions.map((q, i) => (
        <Box key={i} sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
          <Typography>{q.question}</Typography>
          <TextField value={answers[i]} onChange={(e) => handleChange(i, e.target.value)} />
          {results && (
            <Typography color={results[i] ? 'green' : 'red'}>
              {results[i] ? 'Correct' : `Wrong (answer: ${q.answer})`}
            </Typography>
          )}
        </Box>
      ))}
      <Box sx={{ display: 'flex', gap: 2 }}>
        <Button variant="contained" onClick={handleCheck}>Check Answers</Button>
        <Button variant="outlined" onClick={onReset}>Start Over</Button>
      </Box>
    </Box>
  );
}
