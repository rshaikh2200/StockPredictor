"use client";

import React from 'react';
import Image from "next/image";
import getStripe from "@/utils/get-stripe";
import { SignedIn, SignedOut, UserButton } from "@clerk/nextjs";
import {
    AppBar,
    Box,
    Button,
    Container,
    Grid,
    Toolbar,
    Typography,
} from "@mui/material";
import Head from "next/head";
import Link from "next/link";

const HomePage = () => {
  const handleSubmit = async () => {
    try {
      const checkoutSession = await fetch('/api/checkout_sessions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      });

      const checkoutSessionJson = await checkoutSession.json();
  
      const stripe = await getStripe();
      const { error } = await stripe.redirectToCheckout({
        sessionId: checkoutSessionJson.id,
      });
  
      if (error) {
        console.warn(error.message);
      }
    } catch (err) {
      console.error('Error creating checkout session:', err);
    }
  };

  return (
    <Container>
      <Head>
        <title>Flashcard SaaS</title>
      </Head>
      <AppBar position="static">
        <Toolbar>
          <Typography variant="h6" sx={{ flexGrow: 1 }}>
            Flashcard SaaS
          </Typography>
          <Button color="inherit" href="/membership">
            Membership
          </Button>
          <SignedIn>
            <UserButton />
          </SignedIn>
          <SignedOut>
            <Button color="inherit" href="/sign-in">
              Sign In
            </Button>
          </SignedOut>
        </Toolbar>
      </AppBar>
      <Box sx={{ textAlign: 'center', my: 4 }}>
        <Typography variant="h2" component="h1" gutterBottom>
          Welcome to Flashcard SaaS
        </Typography>
        <Typography variant="h5" component="h2" gutterBottom>
          The easiest way to create flashcards from your text.
        </Typography>
        <Button 
          variant="contained" 
          color="primary" 
          sx={{ mt: 2, mr: 2 }} 
          href="/generate"
        >
          Get Started
        </Button>
        <Button 
          variant="outlined" 
          color="primary" 
          sx={{ mt: 2 }} 
          onClick={handleSubmit}
        >
          Learn More
        </Button>
      </Box>
      <Box sx={{ my: 6 }}>
        <Typography variant="h4" component="h2" gutterBottom>
          Features
        </Typography>
        <Grid container spacing={4}>
          {/* Feature items go here */}
        </Grid>
      </Box>
      <Box sx={{ my: 6, textAlign: 'center' }}>
        <Typography variant="h4" component="h2" gutterBottom>
          Pricing
        </Typography>
        <Grid container spacing={4} justifyContent="center">
          {/* Pricing plans go here */}
        </Grid>
      </Box>
    </Container>
  );
};

export default HomePage;
