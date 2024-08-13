// // ImageComponent.js
// import React, { useEffect, useState } from "react";
// import { app } from "../firebase-config";
// import { ref, uploadBytes, getDownloadURL } from "firebase/storage";

// const ImageComponent = () => {
//   const [imageUrl, setImageUrl] = useState("");

//   useEffect(() => {


//     // Reference to the image in Firebase Storage
//     const storageRef = app.storage().ref();
//     const imageRef = storageRef.child("images/1.jpg");

//     // Get the download URL
//     imageRef
//       .getDownloadURL()
//       .then((url) => {
//         // Set the image URL to state
//         setImageUrl(url);
//       })
//       .catch((error) => {
//         console.error("Error getting download URL: ", error);
//       });
//   }, []);

//   return (
//     <div>
//       {imageUrl && (
//         <img
//           src={imageUrl}
//           alt="Firebase Storage Image"
//           style={{ maxWidth: "100%" }}
//         />
//       )}
//     </div>
//   );
// };

// export default ImageComponent;
