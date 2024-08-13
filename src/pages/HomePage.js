import React from "react";
import Header from "../components/Header";
// import Footer from "../components/Footer";
import MainContent from "../components/MainContent";
import Chat from "../components/fachatboat/Chat";
// import PointInPolygon from "../components/PointInPolygon";

function HomePage() {
  return (
    <div className="home_page">
      <Header />
      <MainContent />
      {/* <PointInPolygon /> */}
      {/* <Footer /> */}
    </div>
  );
}

export default HomePage;
