"use client";

import React, { useState, useEffect } from "react";
import { useUser } from "@clerk/nextjs";
import {
  Container,
  Typography,
  Box,
  CircularProgress,
  Grid,
  Card,
  CardContent,
  CardActions,
  Button,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
} from "@mui/material";
import { Check } from "@mui/icons-material";
import getStripe from "@/utils/get-stripe";

const MembershipPage = () => {
  const { user } = useUser();
  const [loading, setLoading] = useState(true);
  const [membership, setMembership] = useState(null);
  const [error, setError] = useState(null);
  const [routerLoaded, setRouterLoaded] = useState(false);

  useEffect(() => {
    if (typeof window !== "undefined") {
      setRouterLoaded(true);
    }

    const fetchMembership = async () => {
      try {
        const res = await fetch("/api/membership-status");
        const data = await res.json();

        if (!res.ok) {
          setError(data);
          return;
        }

        setMembership(data);
      } catch (error) {
        setError("An error occurred while fetching membership status.");
      } finally {
        setLoading(false);
      }
    };

    fetchMembership();
  }, []);

  const handleSelectMembership = async (tier) => {
    if (!routerLoaded) return;

    try {
      if (tier === "premium") {
        const checkoutSession = await fetch("/api/checkout_sessions", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
        });

        const checkoutSessionJson = await checkoutSession.json();
        const stripe = await getStripe();

        const { error } = await stripe.redirectToCheckout({
          sessionId: checkoutSessionJson.id,
        });

        if (error) {
          console.warn(error.message);
        }
      } else {
        const res = await fetch("/api/update-membership", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ tier: "free" }),
        });

        const data = await res.json();
        if (!res.ok) {
          setError(data);
          return;
        }

        setMembership(data);
      }
    } catch (err) {
      console.error("Error processing membership:", err);
    }
  };

  if (loading) {
    return (
      <Container maxWidth="sm" sx={{ textAlign: "center", marginTop: "20vh" }}>
        <CircularProgress />
        <Typography variant="h6">Loading membership status...</Typography>
      </Container>
    );
  }

  if (error) {
    return (
      <Container maxWidth="sm" sx={{ textAlign: "center", marginTop: "20vh" }}>
        <Typography variant="h6" color="error">
          {error}
        </Typography>
      </Container>
    );
  }

  return (
    <Container maxWidth="md" sx={{ textAlign: "center", marginTop: "10vh" }}>
      <Typography variant="h4" gutterBottom>
        Your Membership Status
      </Typography>
      <Typography variant="h6" gutterBottom>
        {membership?.tier === "premium"
          ? "Premium Member"
          : "Free Tier Member"}
      </Typography>
      <Typography variant="body1" gutterBottom>
        Member since: {membership?.startDate}
      </Typography>
      <Box sx={{ my: 6 }}>
        <Typography variant="h4" gutterBottom>
          Upgrad Your Membership
        </Typography>
        <Grid container spacing={4} sx={{ mt: 4 }}>
          <Grid item xs={12} md={6}>
            <PricingCard
              title="Free Tier"
              price="$0.00/month"
              features={[
                "Create 10 Flashcards",
                "Limited Storage Access",
                "Basic Support",
              ]}
              onClick={() => handleSelectMembership("Free Tier")}
            />
          </Grid>
          <Grid item xs={12} md={6}>
            <PricingCard
              title="Gold Tier"
              price="$10.00/month"
              features={[
                "Create Unlimited Flashcards",
                "Full Storage Access",
                "Priority Support",
                "API Integration",
              ]}
              onClick={() => handleSelectMembership("Gold Tier")}
            />
          </Grid>
          <Grid item xs={12} md={6}>
            <PricingCard
              title="Platnium Tier"
              price="$20.00/month"
              features={[
                "Create Unlimited Flashcards",
                "Full Storage Access",
                "Priority Support",
                "API Integration",
              ]}
              onClick={() => handleSelectMembership("Platnium Tier")}
            />
          </Grid>
        </Grid>
      </Box>
    </Container>
  );
};

export default MembershipPage;

const PricingCard = ({ title, price, features, onClick }) => {
  return (
    <Card
      sx={{
        p: 3,
        border: (theme) => `1px solid ${theme.palette.primary.main}`,
        borderRadius: 2,
        backgroundImage: (theme) =>
          `linear-gradient(to bottom, ${theme.palette.primary.light}, ${theme.palette.primary.dark})`,
        textAlign: "center",
      }}
    >
      <CardContent>
        <Typography variant="h5" gutterBottom>
          {title}
        </Typography>
        <Typography color="text.secondary" variant="h6" marginBottom={4}>
          {price}
        </Typography>
        <List>
          {features.map((feature, index) => (
            <ListItem key={index} disablePadding>
              <ListItemIcon>
                <Check />
              </ListItemIcon>
              <ListItemText sx={{ color: "black" }} primary={feature} />
            </ListItem>
          ))}
        </List>
      </CardContent>
      <CardActions>
        <Button variant="contained" fullWidth onClick={onClick}>
          Select
        </Button>
      </CardActions>
    </Card>
  );
};