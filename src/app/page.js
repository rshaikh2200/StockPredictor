"use client";

import React from 'react';
import { Button, Container, Typography, Box } from '@mui/material';
import Head from "next/head";
import Appbar from "/src/app/components/Appbar.jsx"; // Assuming Appbar component exists here

// hi

export default function HomePage() {
  return (
    <Container 
      maxWidth="sm" 
      sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', height: '100vh', justifyContent: 'center' }}
    >
      <Box sx={{ textAlign: 'center', mb: 4 }}>
        <Typography variant="h3" component="h1" gutterBottom>
          Welcome to Our Platform
        </Typography>
        <Typography variant="subtitle1" gutterBottom>
          Get started in testing your knowledge on handling critical workpace situation
        </Typography>
      </Box>

      <Box sx={{ display: 'flex', gap: 2 }}>
        <Button variant="contained" color="primary" size="large">
          Sign In
        </Button>
        <Button variant="outlined" color="secondary" size="large">
          Get Started
        </Button>
      </Box>
    </Container>
  );
}
