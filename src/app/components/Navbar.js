"use client";
import {
  Container,
  Toolbar,
  Typography,
  AppBar,
  Box,
  Button,
  useTheme,
  useMediaQuery,
  Link,
} from "@mui/material";
import { SignedIn, SignedOut, UserButton } from "@clerk/nextjs";

export default function Navbar({ rightContent }) {
  const theme = useTheme();
  const isSmallScreen = useMediaQuery(theme.breakpoints.down("sm"));

  return (
    <Box
      sx={{
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        width: "100%",
        px: isSmallScreen ? 2 : 8,
        bgcolor: "background.default",
        mt: 2,
      }}
    >
      <AppBar
        position="static"
        sx={{
          borderRadius: 10,
          border: "1px solid #ffd6ff",
          width: "100%",
          maxWidth: 1200,
          bgcolor: "background.default",
        }}
      >
        <Toolbar
          sx={{
            display: "flex",
            flexDirection: "row",
            justifyContent: "space-between",

            alignItems: "center",
          }}
        >
          <Typography
            variant={isSmallScreen ? "body1" : "h6"}
            component="div"
            sx={{ flexGrow: 0, flexShrink: 0 }}
          >
            FlashUI
          </Typography>

          {rightContent}
        </Toolbar>
      </AppBar>
    </Box>
  );
}

export function DefaultRightContent() {
  const theme = useTheme();
  const isSmallScreen = useMediaQuery(theme.breakpoints.down("sm"));

  return (
    <>
      <SignedOut>
        <Box
          sx={{
            display: "flex",
            flexDirection: "row",
            gap: { xs: 1, sm: 2, md: 4 },
          }}
        >
          <Button size={isSmallScreen ? "small" : "large"}>Team</Button>
          <Button size={isSmallScreen ? "small" : "large"} href="/sign-in">
            Sign In
          </Button>
        </Box>
      </SignedOut>
      <SignedIn>
        <UserButton />
      </SignedIn>
    </>
  );
}

export function HomeRightContent() {
  const theme = useTheme();
  const isSmallScreen = useMediaQuery(theme.breakpoints.down("sm"));

  return (
    <>
      <SignedIn>
        <Box
          sx={{
            display: "flex",
            flexDirection: "row",
            gap: { xs: 1, sm: 2, md: 4 },
          }}
        >
          <Button size={isSmallScreen ? "small" : "large"} href="/generate">
            Generate
          </Button>
          <Button size={isSmallScreen ? "small" : "large"} href="/flashcards">
            Flashcards
          </Button>
          <UserButton />
        </Box>
      </SignedIn>
    </>
  );
}