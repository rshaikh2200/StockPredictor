"use client";

import { useUser } from "@clerk/nextjs";
import {
    collection,
    doc,
    getDoc,
    writeBatch,
} from "@firebase/firestore";
import {
    Box,
    Button,
    Card,
    CardContent,
    Container,
    Dialog,
    DialogActions,
    DialogContent,
    DialogContentText,
    DialogTitle,
    Grid,
    TextField,
    Typography,
} from "@mui/material";
import { useRouter } from "next/navigation";
import { useState } from "react";

export default function Generate() {
    const [flashcards, setFlashcards] = useState([]);
    const [text, setText] = useState("");
    const [setName, setSetName] = useState("");
    const { user } = useUser(); // Destructure the user object from useUser
    const [dialogOpen, setDialogOpen] = useState(false);

    const handleOpenDialog = () => setDialogOpen(true);
    const handleCloseDialog = () => setDialogOpen(false);

    const handleSubmit = async () => {
        if (!text.trim()) {
            alert('Please enter some text to generate flashcards.');
            return;
        }

        try {
            const response = await fetch('/api/generate', {
                method: 'POST',
                body: JSON.stringify({ text }),
                headers: {
                    'Content-Type': 'application/json',
                },
            });

            if (!response.ok) {
                throw new Error('Failed to generate flashcards');
            }

            const data = await response.json();
            setFlashcards(data);
        } catch (error) {
            console.error('Error generating flashcards:', error);
            alert('An error occurred while generating flashcards. Please try again.');
        }
    };

    const saveFlashcards = async () => {
        if (!setName.trim()) {
            alert('Please enter a name for your flashcard set.');
            return;
        }

        try {
            const userDocRef = doc(collection(db, 'users'), user.id);
            const userDocSnap = await getDoc(userDocRef);

            const batch = writeBatch(db);

            if (userDocSnap.exists()) {
                const userData = userDocSnap.data();
                const updatedSets = [...(userData.flashcardSets || []), { name: setName }];
                batch.update(userDocRef, { flashcardSets: updatedSets });
            } else {
                batch.set(userDocRef, { flashcardSets: [{ name: setName }] });
            }

            const setDocRef = doc(collection(userDocRef, 'flashcardSets'), setName);
            batch.set(setDocRef, { flashcards });

            await batch.commit();

            alert('Flashcards saved successfully!');
            handleCloseDialog();
            setSetName('');
        } catch (error) {
            console.error('Error saving flashcards:', error);
            alert('An error occurred while saving flashcards. Please try again.');
        }
    };

    return (
        <Container maxWidth="md" className="bg-white p-8 rounded-lg shadow-lg">
            <Box sx={{ my: 4 }}>
                <Typography variant="h4" component="h1" gutterBottom className="text-center text-indigo-600">
                    Generate Flashcards
                </Typography>
                <TextField
                    value={text}
                    onChange={(e) => setText(e.target.value)}
                    label="Enter text"
                    fullWidth
                    multiline
                    rows={4}
                    variant="outlined"
                    className="mb-4"
                />
                <Button
                    variant="contained"
                    color="primary"
                    onClick={handleSubmit}
                    fullWidth
                    className="bg-indigo-600 hover:bg-indigo-700"
                >
                    Generate Flashcards
                </Button>
            </Box>

            {flashcards.length > 0 && (
                <Box sx={{ mt: 4 }}>
                    <Typography variant="h5" component="h2" gutterBottom className="text-center text-indigo-600">
                        Generated Flashcards
                    </Typography>
                    <Grid container spacing={2}>
                        {flashcards.map((flashcard, index) => (
                            <Grid item xs={12} sm={6} md={4} key={index}>
                                <Card className="bg-gray-100 border-l-4 border-indigo-600">
                                    <CardContent>
                                        <Typography variant="h6" className="text-indigo-600">Front:</Typography>
                                        <Typography>{flashcard.front}</Typography>
                                        <Typography variant="h6" className="text-indigo-600 mt-2">Back:</Typography>
                                        <Typography>{flashcard.back}</Typography>
                                    </CardContent>
                                </Card>
                            </Grid>
                        ))}
                    </Grid>
                    <Box sx={{ mt: 4, display: 'flex', justifyContent: 'center' }}>
                        <Button 
                            variant="contained" 
                            color="primary" 
                            onClick={handleOpenDialog} 
                            className="bg-indigo-600 hover:bg-indigo-700"
                        >
                            Save Flashcards
                        </Button>
                    </Box>
                </Box>
            )}

            <Dialog open={dialogOpen} onClose={handleCloseDialog}>
                <DialogTitle className="text-indigo-600">Save Flashcard Set</DialogTitle>
                <DialogContent>
                    <DialogContentText>
                        Please enter a name for your flashcard set.
                    </DialogContentText>
                    <TextField
                        autoFocus
                        margin="dense"
                        label="Set Name"
                        type="text"
                        fullWidth
                        value={setName}
                        onChange={(e) => setSetName(e.target.value)}
                        className="mt-2"
                    />
                </DialogContent>
                <DialogActions>
                    <Button onClick={handleCloseDialog} className="text-indigo-600">Cancel</Button>
                    <Button onClick={saveFlashcards} color="primary" className="bg-indigo-600 hover:bg-indigo-700">
                        Save
                    </Button>
                </DialogActions>
            </Dialog>
        </Container>
    );
}
