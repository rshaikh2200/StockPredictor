'use client'
import Image from "next/image";
import getStripe from "@/utils/get_stripe";
import { SignIn, SignedOut, UserButton } from "@clerk/nextjs";
import { AppBar, Container, Toolbar, Typography, Button, Link, IconButton, Grid, Avatar, Box, Paper } from "@mui/material";
import MenuIcon from "@mui/icons-material/Menu";
import { CheckCircle as CheckCircleIcon, GitHub } from '@mui/icons-material';
import FeatureIcon1 from '@mui/icons-material/FlashOn';
import FeatureIcon2 from '@mui/icons-material/Storage';
import FeatureIcon3 from '@mui/icons-material/Security';
import FacebookIcon from '@mui/icons-material/Facebook'; // Import social icons
import GitHubIcon from '@mui/icons-material/GitHub';
import LinkedInIcon from '@mui/icons-material/LinkedIn';
import 'slick-carousel/slick/slick.css';
import 'slick-carousel/slick/slick-theme.css';
import Slider from 'react-slick';

export default function Home() {
  const handleSubmit = async() =>{
    const checkoutSession = await fetch('api/checkout_session',{
      method: 'POST',
      headers: {
        origin: 'http://localhost:3000',
    },
  })
  const checkoutSessionJson = await checkoutSession.json()

  if(checkoutSession.statusCode === 500){
    console.error(checkoutSession.message)
    return
  }
  const stripe = await getStripe();
  const { error } = await stripe.redirectToCheckout({
    sessionId: checkoutSessionJson.id,
  })
  if(error){
    console.warn(error.message)
    return
  }
}

const settings = {
  dots: true, // Show navigation dots
  infinite: true, // Infinite loop
  speed: 500, // Transition speed
  slidesToShow: 3, // Show 3 testimonials at a time
  slidesToScroll: 1, // Scroll one testimonial at a time
  arrows: true, // Show navigation arrows
  responsive: [
    {
      breakpoint: 1024,
      settings: {
        slidesToShow: 2, // Show 2 testimonials on medium screens
        slidesToScroll: 1,
        infinite: true,
        dots: true,
      },
    },
    {
      breakpoint: 600,
      settings: {
        slidesToShow: 1, // Show 1 testimonial on small screens
        slidesToScroll: 1,
      },
    },
  ],
};

const teamMembers = [
  {
    name: 'Rimsha Ataullah',
    position: 'AI Engineer & Data Science Enthusiast ',
    bio: 'An AI and data science specialist with a solid computer science background, focused on delivering innovative solutions and impactful results.',
    image: 'images/rimsha.jpg', // Replace with actual image paths
    facebook: 'https://www.facebook.com/rimshaataullah?mibextid=ZbWKwL',
    GitHub: 'https://github.com/Rims99',
    linkedin: 'https://www.linkedin.com/in/rimsha-ataullah/',
  },
  {
    name: 'Syeda Eman',
    position: 'AI Engineer',
    bio: 'An aspiring AI Engineer and tech enthusiast who is passionate about using AI to address real-world challenges and create meaningful solutions.',
    image: 'images/eman.jpg',
    facebook: 'https://www.facebook.com/profile.php?id=100095029180487',
    GitHub: 'https://github.com/Syeda-Eman',
    linkedin: 'https://www.linkedin.com/in/syeda-eman/',
  },
  {
    name: 'Kinza Syed',
    position: 'AI Engineer',
    bio: 'A dedicated AI enthusiast with extensive experience in developing AI-driven solutions who focuses on creating advanced AI applications',
    image: 'images/kinza.jpg',
    facebook: 'https://www.facebook.com/profile.php?id=100094668984112',
    GitHub: 'https://github.com/KinzaSyedHussain',
    linkedin: 'https://www.linkedin.com/in/kinza-syed-1139ba262/',
  },
  // Add more team members as needed
];

  return (
    <>
      {/* Navigation Bar (App Bar) */}
      <AppBar position="static" sx={{ backgroundColor: '#000000', color: '#FFFFFF' }}>
        <Toolbar>
          <IconButton edge="start" color="inherit" aria-label="menu" sx={{ mr: 2 }}>
            <MenuIcon />
          </IconButton>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            Galaxy Flashcards
          </Typography>
          <Button color="inherit" href="/sign_in">Sign In</Button>
          <Button color="inherit" href="/sign_up">Sign Up</Button>
          <UserButton />
        </Toolbar>
      </AppBar>

  <Box
  sx={{
    backgroundImage: 'url(/images/1.png)', // Provide the correct path to the image
    backgroundSize: 'cover',
    backgroundPosition: 'center',
    color: '#FFFFFF',
    py: 8,
    height: '80vh',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'flex-end', // Align content to the right
    position: 'relative',
    textAlign: 'left', // Align text to the right
    overflow: 'hidden',
    pr: 4, // Add some padding to the right
  }}
>
  <Container maxWidth="lg">
    <Box>
      <Typography
        variant="h2"
        gutterBottom
        sx={{
          animation: 'fadeIn 2s ease-in-out', 
          animationDelay: '0.5s', // Apply fade-in animation
        }}
      >
        Galaxy Flashcards
      </Typography>
      <Typography
        variant="h5"
        paragraph
        sx={{
          animation: 'fadeIn 2s ease-in-out',
          animationDelay: '0.5s', // Delayed animation for subtitle
        }}
      >
        Create, Engage, learn, and excel effortlessly
      </Typography>
      <Button
        variant="contained"
        sx={{
          backgroundColor: '#104EB1',
          mt: 4,
        }}
        href="/generate"
      >
        Get Started
      </Button>
    </Box>
  </Container>
</Box>

      {/* Features Section */}
      <Box sx={{ py: 8, backgroundColor: '#000000', color: '#FFFFFF' }}>
        <Container maxWidth="lg">
          {/* Feature Section Heading */}
          <Box sx={{ textAlign: 'center', mb: 6 }}>
            <Typography variant="h3" gutterBottom>
              Key Features
            </Typography>
            <Typography variant="h6">
              Discover what makes our Flashcard SaaS stand out.
            </Typography>
          </Box>

          {/* Feature Grid */}
          <Grid container spacing={4}>
            {/* Feature 1 */}
            <Grid item xs={12} md={4}>
              <Box
                sx={{
                  backgroundColor: '#333333',
                  color: '#FFFFFF',
                  p: 4,
                  borderRadius: 2,
                  textAlign: 'center',
                  boxShadow: 3,
                }}
              >
                <FeatureIcon1 sx={{ fontSize: 60, color: '#104EB1' }} />
                <Typography variant="h5" sx={{ mt: 2 }}>
                  Lightning Fast
                </Typography>
                <Typography variant="body1" sx={{ mt: 1 }}>
                  Experience incredibly fast load times and smooth performance.
                </Typography>
              </Box>
            </Grid>

            {/* Feature 2 */}
            <Grid item xs={12} md={4}>
              <Box
                sx={{
                  backgroundColor: '#333333',
                  color: '#FFFFFF',
                  p: 4,
                  borderRadius: 2,
                  textAlign: 'center',
                  boxShadow: 3,
                }}
              >
                <FeatureIcon2 sx={{ fontSize: 60, color: '#104EB1' }} />
                <Typography variant="h5" sx={{ mt: 2 }}>
                  Secure Storage
                </Typography>
                <Typography variant="body1" sx={{ mt: 1 }}>
                  Your data is safe with industry-leading encryption and security practices.
                </Typography>
              </Box>
            </Grid>

            {/* Feature 3 */}
            <Grid item xs={12} md={4}>
              <Box
                sx={{
                  backgroundColor: '#333333',
                  color: '#FFFFFF',
                  p: 4,
                  borderRadius: 2,
                  textAlign: 'center',
                  boxShadow: 3,
                }}
              >
                <FeatureIcon3 sx={{ fontSize: 60, color: '#104EB1' }} />
                <Typography variant="h5" sx={{ mt: 2 }}>
                  Reliable Support
                </Typography>
                <Typography variant="body1" sx={{ mt: 1 }}>
                  Our team is here to help you 24/7 with any questions or issues.
                </Typography>
              </Box>
            </Grid>
          </Grid>
        </Container>
      </Box>

      {/* Pricing Section */}
      <Box sx={{ py: 8, backgroundColor: '#111111', color: '#FFFFFF' }}>
        <Container maxWidth="lg">
          <Box sx={{ textAlign: 'center', mb: 6 }}>
            <Typography variant="h3" gutterBottom>
              Pricing Plans
            </Typography>
            <Typography variant="h6">
              Choose a plan that fits your needs.
            </Typography>
          </Box>

          {/* Pricing Grid */}
          <Grid container spacing={4}>
            {/* Basic Plan */}
            <Grid item xs={12} md={4}>
              <Paper
                elevation={3}
                sx={{
                  p: 4,
                  borderRadius: 2,
                  backgroundColor: '#222222',
                  textAlign: 'center',
                  color: '#FFFFFF',
                }}
              >
                <Typography variant="h4" gutterBottom>
                  Basic
                </Typography>
                <Typography variant="h5" gutterBottom>
                  $10/month
                </Typography>
                <Typography variant="body1" paragraph>
                  Perfect for individual users. Get access to all basic features.
                </Typography>
                <Button variant="contained"
  sx={{
    backgroundColor: '#104EB1', // Change the button background color here
    color: '#ffffff',           // Change the button text color here
    '&:hover': {
      backgroundColor: '#104EB1', // Change the hover background color here
    },
  }} onClick={handleSubmit}>
                  Subscribe
                </Button>
              </Paper>
            </Grid>

            {/* Standard Plan */}
            <Grid item xs={12} md={4}>
              <Paper
                elevation={3}
                sx={{
                  p: 4,
                  borderRadius: 2,
                  backgroundColor: '#222222',
                  textAlign: 'center',
                  color: '#FFFFFF',
                }}
              >
                <Typography variant="h4" gutterBottom>
                  Standard
                </Typography>
                <Typography variant="h5" gutterBottom>
                  $20/month
                </Typography>
                <Typography variant="body1" paragraph>
                  Ideal for small teams. Includes all Basic features plus advanced options.
                </Typography>
                <Button variant="contained"
  sx={{
    backgroundColor: '#104EB1', // Change the button background color here
    color: '#ffffff',           // Change the button text color here
    '&:hover': {
      backgroundColor: '#104EB1', // Change the hover background color here
    },
  }} onClick={handleSubmit}>
                  Subscribe
                </Button>
              </Paper>
            </Grid>

            {/* Premium Plan */}
            <Grid item xs={12} md={4}>
              <Paper
                elevation={3}
                sx={{
                  p: 4,
                  borderRadius: 2,
                  backgroundColor: '#222222',
                  textAlign: 'center',
                  color: '#FFFFFF',
                }}
              >
                <Typography variant="h4" gutterBottom>
                  Premium
                </Typography>
                <Typography variant="h5" gutterBottom>
                  $30/month
                </Typography>
                <Typography variant="body1" paragraph>
                  For larger organizations. Access to all features and premium support.
                </Typography>
                <Button variant="contained"
  sx={{
    backgroundColor: '#104EB1', // Change the button background color here
    color: '#ffffff',           // Change the button text color here
    '&:hover': {
      backgroundColor: '#104EB1', // Change the hover background color here
    },
  }} onClick={handleSubmit}>
                  Subscribe
                </Button>
              </Paper>
            </Grid>
          </Grid>
        </Container>
      </Box>

      {/* Meet Our Team Section */}
      <Box sx={{ py: 8, backgroundColor: '#000000', color: '#FFFFFF' }}>
        <Container maxWidth="lg">
          <Box sx={{ textAlign: 'center', mb: 6 }}>
            <Typography variant="h3" gutterBottom>
              Meet Our Team
            </Typography>
            <Typography variant="h6">
              Get to know the passionate individuals behind Galaxy Flashcards.
            </Typography>
          </Box>

          {/* Team Grid */}
          <Grid container spacing={4}>
            {teamMembers.map((member) => (
              <Grid item xs={12} md={4} key={member.name}>
                <Box
                  sx={{
                    backgroundColor: '#222222',
                    color: '#FFFFFF',
                    p: 4,
                    borderRadius: 2,
                    textAlign: 'center',
                    boxShadow: 3,
                  }}
                >
                  <Avatar
                    src={member.image}
                    alt={member.name}
                    sx={{ width: 120, height: 120, mx: 'auto' }}
                  />
                  <Typography variant="h5" sx={{ mt: 2 }}>
                    {member.name}
                  </Typography>
                  <Typography variant="body1" sx={{ mt: 1 }}>
                    {member.position}
                  </Typography>
                  <Typography variant="body2" sx={{ mt: 2 }}>
                    {member.bio}
                  </Typography>
                  <Box sx={{ mt: 2 }}>
                    <Link href={member.facebook} target="_blank" rel="noopener">
                      <FacebookIcon sx={{ color: '#FFFFFF', mx: 1 }} />
                    </Link>
                    <Link href={member.GitHub} target="_blank" rel="noopener">
                      <GitHub sx={{ color: '#FFFFFF', mx: 1 }} />
                    </Link>
                    <Link href={member.linkedin} target="_blank" rel="noopener">
                      <LinkedInIcon sx={{ color: '#FFFFFF', mx: 1 }} />
                    </Link>
                  </Box>
                </Box>
              </Grid>
            ))}
          </Grid>
        </Container>
      </Box>

      {/* Testimonial Section */}
<Box sx={{ py: 8, backgroundColor: '#111111', color: '#FFFFFF' }}>
  <Container maxWidth="lg">
    <Box sx={{ textAlign: 'center', mb: 6 }}>
      <Typography variant="h3" gutterBottom>
        What Our Users Say
      </Typography>
      <Typography variant="h6">
        Hear from those who have experienced the Galaxy Flashcards difference.
      </Typography>
    </Box>

    {/* Testimonials Slider */}
    <Slider {...settings}>
      <Box 
        sx={{
          backgroundColor: '#222222',
          color: '#FFFFFF',
          p: 6,
          borderRadius: 2,
          textAlign: 'center',
          boxShadow: 3,
        }}
      >
        <Typography variant="h6" paragraph>
          "Galaxy Flashcards has transformed the way I study. The user interface is incredibly intuitive, and the flashcards are a game-changer."
        </Typography>
        <Typography variant="subtitle1">- Jane Doe</Typography>
      </Box>
      <Box 
        sx={{
          backgroundColor: '#222222',
          color: '#FFFFFF',
          p: 6,
          borderRadius: 2,
          textAlign: 'center',
          boxShadow: 3,
        }}
      >
        <Typography variant="h6" paragraph>
          "The personalized flashcards and seamless integration with my study schedule have made a huge difference in my academic performance."
        </Typography>
        <Typography variant="subtitle1">- John Smith</Typography>
      </Box>
      <Box 
        sx={{
          backgroundColor: '#222222',
          color: '#FFFFFF',
          p: 6,
          borderRadius: 2,
          textAlign: 'center',
          boxShadow: 3,
        }}
      >
        <Typography variant="h6" paragraph>
          "I love the variety of features and the ease of use. Galaxy Flashcards is a must-have tool for anyone serious about studying."
        </Typography>
        <Typography variant="subtitle1">- Emily Johnson</Typography>
      </Box>
    </Slider>
  </Container>
</Box>

      <Box
      sx={{
        backgroundColor: '#000000',
        color: '#FFFFFF',
        py: 2,
        textAlign: 'center'
      }}
    >
      <Container maxWidth="lg">
        <Typography variant="body2">
          © {new Date().getFullYear()} Galaxy Flashcards. All Rights Reserved.
        </Typography>
      </Container>
    </Box>
    </>
  );
}
