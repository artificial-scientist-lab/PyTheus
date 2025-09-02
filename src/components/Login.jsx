import React from 'react';
import { Box, TextField, Button, Typography } from '@mui/material';

export default function Login() {
  const handleSubmit = (e) => {
    e.preventDefault();
    alert('Logged in!');
  };

  return (
    <Box component="form" onSubmit={handleSubmit} sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      <Typography variant="h4" align="center">Login</Typography>
      <TextField label="Username" required />
      <TextField label="Password" type="password" required />
      <Button type="submit" variant="contained">Login</Button>
    </Box>
  );
}
