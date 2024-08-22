import { Box } from "@mui/material";
import { styled } from "@mui/system";
import anime from "animejs";
import { useState } from "react";

const CardContainer = styled(Box)({
  height: "220px",
  borderRadius: "10px",
});

const Card = styled(Box)({
  position: "relative",
  height: "100%",
  width: "100%",
  borderRadius: "10px",
  transformStyle: "preserve-3d",
  transform: "rotateY(0deg)",
  transition: "transform 0.6s",
});

const Front = styled(Box)({
  display: "flex",
  justifyContent: "center",
  alignItems: "center",
  width: "100%",
  height: "100%",
  borderRadius: "10px",
  backfaceVisibility: "hidden",
  position: "absolute",
  fontSize: "1rem",
  fontWeight: 500,
  padding: "1rem",
  backgroundColor: "#E0E0E0", // Replace with your desired color
  color: "#000000", // Replace with your desired color
});

const Back = styled(Box)({
  display: "flex",
  justifyContent: "center",
  alignItems: "center",
  width: "100%",
  height: "100%",
  borderRadius: "10px",
  backfaceVisibility: "hidden",
  position: "absolute",
  top: 0,
  left: 0,
  fontSize: "1rem",
  fontWeight: 500,
  padding: "1rem",
  transform: "rotateY(180deg)",
  backgroundColor: "#B0B0B0", // Replace with your desired color
  color: "#FFFFFF", // Replace with your desired color
  textShadow: "1px 2px 3px rgba(0, 0, 0, 0.2)",
});

const FlipCard = ({ flashcard, index }) => {
  const [playing, setPlaying] = useState(false);

  const handleClick = () => {
    if (playing) return;
    setPlaying(true);

    anime({
      targets: `.card-${index}`,
      scale: [{ value: 1 }, { value: 1.4 }, { value: 1, delay: 100 }],
      rotateY: { value: "+=180", delay: 100 },
      easing: "easeInOutSine",
      duration: 200,
      complete: function () {
        setPlaying(false);
      },
    });
  };

  return (
    <CardContainer onClick={handleClick} aria-readonly>
      <Card className={`card-${index}`}>
        <Front className="front">{flashcard.front}</Front>
        <Back className="back">{flashcard.back}</Back>
      </Card>
    </CardContainer>
  );
};

export default FlipCard;
