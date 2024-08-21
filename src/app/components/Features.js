import {
    Typography,
    Box,
    Button,
    useTheme,
    useMediaQuery,
    Card,
    CardContent,
    Grid,
    Container,
  } from "@mui/material";
  
  export default function Features() {
    const theme = useTheme();
    const isSmallScreen = useMediaQuery(theme.breakpoints.down("sm"));
    const features = [
      {
        title: "AI-Powered Flashcards",
        description:
          "Create flashcards instantly with our advanced AI technology.",
      },
      {
        title: "Turn Your Notes into Flashcards",
        description:
          "Upload your PDF notes, and let our AI convert them into customized flashcards.",
      },
      {
        title: "Create and Manage Flashcard Lists",
        description:
          "Keep your study sessions organized by categorizing flashcards into lists.",
      },
      {
        title: "Focused Learning with Category-Specific Flashcards",
        description:
          "Study flashcards by category to focus on specific styles/libraries.",
      },
    ];
  
    return (
      <Box sx={{ py: 8 }}>
        <Container maxWidth="lg">
          <Typography
            variant="h3"
            align="center"
            gutterBottom
            color="secondary.main"
            sx={{ mb: 4 }}
          >
            Features
          </Typography>
          <Grid container spacing={isSmallScreen ? 2 : 4} justifyContent="center">
            {features.map((feature, index) => (
              <Grid item key={index} xs={12} sm={12} md={6}>
                <Card
                  sx={{
                    height: "100%",
                    display: "flex",
                    flexDirection: "column",
                    transition:
                      "transform 0.3s ease-in-out, box-shadow 0.3s ease-in-out",
                    "&:hover": {
                      transform: "translateY(-5px)",
                      boxShadow: 3,
                    },
                  }}
                >
                  <CardContent sx={{ flexGrow: 1, p: { xs: 2, sm: 3 } }}>
                    <Typography gutterBottom variant="h5" component="h2">
                      {feature.title}
                    </Typography>
                    <Typography color="white">{feature.description}</Typography>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </Container>
      </Box>
    );
  }