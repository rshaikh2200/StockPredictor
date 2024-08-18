"use client";

import { useUser } from "@clerk/nextjs";
import {
    collection,
    doc,
    getDoc,
    writeBatch,
    setDoc,
} from "@firebase/firestore";
import {
    Box,
    Button,
    Card,
    CardActionArea,
    CardContent,
    Container,
    Dialog,
    DialogActions,
    DialogContent,
    DialogContentText,
    DialogTitle,
    Grid,
    Paper,
    TextField,
    Typography,
} from "@mui/material";
import { useRouter } from "next/navigation";
import { useState } from "react";


export default function Generate() {
    const [flashcards, setFlashcards] = useState([]);
    const [flipped, setFlipped] = useState([]);
    const [text, setText] = useState("");
    const [name, setName] = useState("");
    const [open, setOpen] = useState(false);
    const { user } = useUser(); // Destructure the user object from useUser
    const router = useRouter();

    const handleSubmit = async () => {
        try {
            const res = await fetch("/api/generate", {
                method: "POST",
                body: text,
            });

            if (!res.ok) {
                throw new Error("Failed to fetch flashcards");
            }

            const data = await res.json();

            if (!data) {
                throw new Error("No data returned");
            }

            setFlashcards(data);
        } catch (error) {
            console.error("Error fetching flashcards:", error);
            alert(
                "There was an error generating the flashcards. Please try again.",
            );
        }
    };

    const handleCardClick = (id) => {
        setFlipped((prev) => ({
            ...prev,
            [id]: !prev[id],
        }));
    };

    const handleOpen = () => {
        setOpen(true);
    };

    const handleClose = () => {
        setOpen(false);
    };

    const saveFlashcard = async () => {
        if (!name) {
            alert("Please enter a name");
            return;
        }

        if (!user) {
            alert("User is not authenticated");
            return;
        }

        const batch = writeBatch(db);
        const userDocRef = doc(collection(db, "users"), user.id); // Use user.id to get the correct document reference
        const docSnap = await getDoc(userDocRef);

        if (docSnap.exists()) {
            const collections = docSnap.data().flashcards || [];
            if (collections.find((f) => f.name === name)) {
                alert("Flashcard collection with the same name already exists");
                return;
            } else {
                collections.push({ name });
                batch.set(
                    userDocRef,
                    { flashcards: collections },
                    { merge: true },
                );
            }
        } else {
            batch.set(userDocRef, { flashcards: [{ name }] });
        }

        const colRef = collection(userDocRef, name);
        flashcards.forEach((flashcard) => {
            const cardDocRef = doc(colRef);
            batch.set(cardDocRef, flashcard);
        });

        await batch.commit();
        handleClose();
        router.push("/flashcards");
    };

    return (
        <Container maxWidth="md">
            <Button href="/flashcards" variant="contained" color="primary">
                Check Saved Flashcards
            </Button>
            <Box
                sx={{
                    mt: 4,
                    mb: 6,
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "center",
                }}
            >
                <Typography variant="h4">Generate Flashcards</Typography>
                <Paper sx={{ p: 4, width: "100%" }}>
                    <TextField
                        value={text}
                        onChange={(e) => setText(e.target.value)}
                        label="Enter text"
                        fullWidth
                        multiline
                        rows={4}
                        variant="outlined"
                        sx={{ mb: 2 }}
                    />
                    <Button
                        variant="contained"
                        color="primary"
                        onClick={handleSubmit}
                        fullWidth
                    >
                        Submit
                    </Button>
                </Paper>
            </Box>

            {flashcards.length > 0 && (
                <Box sx={{ mt: 4 }}>
                    <Typography variant="h5">Flashcards Preview</Typography>
                    <Grid container spacing={3}>
                        {flashcards.map((flashcard, index) => (
                            <Grid item xs={12} sm={6} md={4} key={index}>
                                <Card>
                                    <CardActionArea
                                        onClick={() => handleCardClick(index)}
                                    >
                                        <CardContent>
                                            <Box
                                                sx={{
                                                    perspective: "1000px",
                                                    "& > div": {
                                                        transition:
                                                            "transform 0.6s",
                                                        transformStyle:
                                                            "preserve-3d",
                                                        position: "relative",
                                                        width: "100%",
                                                        height: "200px",
                                                        boxShadow:
                                                            "0 4px 8px 0 rgba(0, 0, 0, 0.2)",
                                                        transform: flipped[
                                                            index
                                                        ]
                                                            ? "rotateY(180deg)"
                                                            : "rotateY(0deg)",
                                                    },
                                                    "& > div > div": {
                                                        transition:
                                                            "transform 0.6s",
                                                        transformStyle:
                                                            "preserve-3d",
                                                        position: "absolute",
                                                        width: "100%",
                                                        height: "100%",
                                                        backfaceVisibility:
                                                            "hidden",
                                                        display: "flex",
                                                        alignItems: "center",
                                                        justifyContent:
                                                            "center",
                                                        padding: 2,
                                                        boxSizing: "border-box",
                                                    },
                                                }}
                                            >
                                                <div>
                                                    {/* Front Side */}
                                                    <div
                                                        style={{
                                                            transform:
                                                                "rotateY(0deg)",
                                                        }}
                                                    >
                                                        <Typography
                                                            variant="h5"
                                                            component="div"
                                                        >
                                                            {flashcard.front}
                                                        </Typography>
                                                    </div>

                                                    {/* Back Side */}
                                                    <div
                                                        style={{
                                                            transform:
                                                                "rotateY(180deg)",
                                                        }}
                                                    >
                                                        <Typography
                                                            variant="h5"
                                                            component="div"
                                                        >
                                                            {flashcard.back}
                                                        </Typography>
                                                    </div>
                                                </div>
                                            </Box>
                                        </CardContent>
                                    </CardActionArea>
                                </Card>
                            </Grid>
                        ))}
                    </Grid>

                    <Box
                        sx={{
                            mt: 4,
                            display: "flex",
                            justifyContent: "center",
                        }}
                    >
                        <Button
                            variant="contained"
                            color="primary"
                            onClick={handleOpen}
                        >
                            Save
                        </Button>
                    </Box>
                </Box>
            )}

            <Dialog open={open} onClose={handleClose}>
                <DialogTitle>Save Flashcards</DialogTitle>
                <DialogContent>
                    <DialogContentText>
                        Enter a name for the flashcard collection
                    </DialogContentText>
                    <TextField
                        autoFocus
                        margin="dense"
                        label="Collection Name"
                        type="text"
                        fullWidth
                        value={name}
                        onChange={(e) => setName(e.target.value)}
                        variant="outlined"
                    />
                </DialogContent>
                <DialogActions>
                    <Button onClick={handleClose} color="primary">
                        Cancel
                    </Button>
                    <Button onClick={saveFlashcard} color="primary">
                        Save
                    </Button>
                </DialogActions>
            </Dialog>
        </Container>
    );
}
