import React from "react";
import cwclogo from "../images/cwclogo.gif";
// import emblem_india from "../images/emblem-india.gif";
import ministry_panel from "../images/ministry-panel_1_1.png";

function Header() {
  return (
    <div className="header">
      <div className="header_left">
        <div className="header_left_image">
          <img src={cwclogo} alt="logo_right"/>
        </div>

        <div className="header_left_titles">
          <h1>Central Water Commission</h1>
          <h3>(Serving the nation since 1945)</h3>
        </div>
      </div>

      <div className="header_right">
        <div className="header_right_image">
          <img src={ministry_panel} alt="logo_right"/>
        </div>
      </div>
    </div>
  );
}

export default Header;
