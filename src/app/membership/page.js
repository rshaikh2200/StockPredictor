"use client";

import React, { useState, useEffect } from 'react';
import { useUser } from "@clerk/nextjs";
import { Container, Typography, Box, Button, CircularProgress } from "@mui/material";
import { useRouter } from "next/router";
import getStripe from "@/utils/get-stripe";

const MembershipPage = () => {
    const { user } = useUser();
    const router = useRouter();
    const [loading, setLoading] = useState(true);
    const [membership, setMembership] = useState(null);
    const [error, setError] = useState(null);

    useEffect(() => {
        const fetchMembership = async () => {
            try {
                const res = await fetch('/api/membership-status');
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
        if (tier === 'premium') {
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
        } else {
            // Handle free tier selection
            try {
                const res = await fetch('/api/update-membership', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ tier: 'free' })
                });

                const data = await res.json();
                if (!res.ok) {
                    setError(data);
                    return;
                }

                setMembership(data);
            } catch (err) {
                console.error('Error updating membership:', err);
            }
        }
    };

    if (loading) {
        return (
            <Container maxWidth="sm" sx={{ textAlign: 'center', marginTop: '20vh' }}>
                <CircularProgress />
                <Typography variant="h6">Loading membership status...</Typography>
            </Container>
        );
    }

    if (error) {
        return (
            <Container maxWidth="sm" sx={{ textAlign: 'center', marginTop: '20vh' }}>
                <Typography variant="h6" color="error">{error}</Typography>
            </Container>
        );
    }

    return (
        <Container maxWidth="sm" sx={{ textAlign: 'center', marginTop: '20vh' }}>
            <Typography variant="h4" gutterBottom>
                Membership Status
            </Typography>
            <Typography variant="h6" gutterBottom>
                {membership?.tier === 'premium' ? 'Premium Member' : 'Free Tier Member'}
            </Typography>
            <Typography variant="body1" gutterBottom>
                Member since: {membership?.startDate}
            </Typography>
            <Box sx={{ mt: 4 }}>
                <Button
                    variant="contained"
                    color="primary"
                    sx={{ mr: 2 }}
                    onClick={() => handleSelectMembership('free')}
                >
                    Free Tier
                </Button>
                <Button
                    variant="outlined"
                    color="secondary"
                    onClick={() => handleSelectMembership('premium')}
                >
                    Premium Tier - $10/month
                </Button>
            </Box>
        </Container>
    );
};

export default MembershipPage;
