import React from "react";
import Navigation from "./Navigation.jsx";

const Layout = ({ children }) => {
  return (
    <div className="app-container">
      <Navigation />
      <main className="app-content">{children}</main>
    </div>
  );
};

export default Layout;
