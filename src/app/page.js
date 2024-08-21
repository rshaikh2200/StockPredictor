"use client";

import React from 'react';
import getStripe from "@/utils/get-stripe";
import { SignedIn, SignedOut, UserButton } from "@clerk/nextjs";
import {
    AppBar,
    Box,
    Button,
    Container,
    Toolbar,
    Typography,
} from "@mui/material";
import Head from "next/head";
import Link from "next/link";

// Define Navbar component
const Navbar = ({ rightContent }) => (
  <AppBar position="static">
    <Toolbar>
      <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
        Flashcards Saas
      </Typography>
      {rightContent}
    </Toolbar>
  </AppBar>
);

// Define DefaultRightContent component (if necessary)
const DefaultRightContent = () => (
  <Box sx={{ display: 'flex', alignItems: 'center' }}>
    <SignedIn>
      <UserButton />
    </SignedIn>
    <SignedOut>
      <Button color="inherit" component={Link} href="/sign-in">
        Sign In
      </Button>
    </SignedOut>
  </Box>
);

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
    <Box
      sx={{
        display: "flex",
        flexDirection: "column",
        minHeight: "100vh",
        overflow: "hidden",
        bgcolor: "background.default",
      }}
    >
      <Navbar rightContent={<DefaultRightContent />} />
      <Head>
        <title>Flashcards Saas</title>
        <meta property="description" content="Flashcards created with AI" />
      </Head>

      <Box
        sx={{
          display: "flex",
          flexGrow: 1,
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          mt: 4,
        }}
      >
        <Typography
          variant="h3"
          align="center"
          color="secondary.main"
          sx={{
            fontWeight: "bold",
            transform: "translateZ(0)",
            transition: "transform 0.7s ease-out",
            "&:hover": {
              transform: "translateY(-10px) translateZ(0)",
            },
            mb: 2,
          }}
        >
          FlashUI
        </Typography>
        <Typography
          variant="h6"
          align="center"
          color="tertiary.main"
        >
          Supercharge Your UI Learning with AI-Powered Flashcards
        </Typography>
        <Button
          variant="contained"
          color="primary"
          sx={{ mt: 6 }}
          href="/sign-up"
        >
          Get Started
        </Button>

        {/* Features */}
        <Features />

        {/* Pricing */}
        <Container
          maxWidth="md"
          sx={{
            display: "flex",
            flexDirection: "column",
            py: 8,
          }}
        >
          <Typography
            variant="h3"
            align="center"
            gutterBottom
            color="secondary.main"
            sx={{ mb: 4 }}
          >
            Pricing
          </Typography>
          <Box
            sx={{
              display: "flex",
              flexDirection: "column",
              justifyContent: "space-between",
            }}
          >
            <Box>
              <Typography variant="h4">
                Get Started with FlashUI for Only $1/Month
              </Typography>
              <Typography variant="h5" color="white">
                Only for a limited time
              </Typography>
              <Typography variant="h6" color="tertiary.main" sx={{ mt: 4 }}>
                &quot;Simple, Affordable, Effective&quot;
              </Typography>
              <Typography variant="body1" color="white">
                For just $1 a month, gain access to all the powerful features of
                FlashUI. Enhance your UI skills without breaking the bank.
              </Typography>
              <Button
                href="/sign-up"
                variant="contained"
                color="primary"
                sx={{ mt: 6 }}
              >
                Get Started
              </Button>
            </Box>
          </Box>
        </Container>
      </Box>
    </Box>
  );
}

export default HomePage;
