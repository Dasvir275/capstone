// src/components/Chatbot.js
import React, { useState } from "react";
import "./Chatbot.css";

const Chatbot = () => {
  const [isOpen, setIsOpen] = useState(false);

  const toggleChatbot = () => {
    setIsOpen(!isOpen);
  };

  const faqs = [
    { question: "What is flood inundation?", answer: "Flood inundation refers to the flooding of normally dry areas." },
    { question: "How can I prepare for a flood?", answer: "To prepare for a flood, ensure you have an emergency plan and supplies." },
    { question: "What should I do during a flood?", answer: "During a flood, move to higher ground and follow local authorities' instructions." },
    // Add more FAQs here
  ];

  return (
    <div className="chatbot-container">
      <button className="chatbot-toggle" onClick={toggleChatbot}>
        {isOpen ? "Close Chatbot" : "Chat with us"}
      </button>
      {isOpen && (
        <div className="chatbot-content">
          <h3>Flood Inundation FAQ</h3>
          <ul>
            {faqs.map((faq, index) => (
              <li key={index}>
                <strong>{faq.question}</strong>
                <p>{faq.answer}</p>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default Chatbot;
