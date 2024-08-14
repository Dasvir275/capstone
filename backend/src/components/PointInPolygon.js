// import React, { useState } from 'react';

// const PointInPolygon = () => {
//   const [polygon, setPolygon] = useState([
//     [90.98953316168216, 26.323165319486552],
//     [91.02395126776126, 26.323165319486552],
//     [91.02395126776126, 26.342165634762075],
//     [90.98953316168216, 26.342165634762075],
//   ]);

//   const hospitals = [
//     {
//       name: 'Barpeta Civil Hospital, Kalgachia',
//       lat: 26.3556374,
//       lng: 90.8679944,
//       address: 'Civil Hospital, Road, Kalgachia, Assam 781319, India',
//     },
//     {
//       name: 'Sanjivani Hospital',
//       lat: 26.3252586,
//       lng: 90.9986036,
//       address: '8XGX+4C5, Barpeta-Hospital-Jania Rd, Muslimpatty, Barpeta, Assam 781301, India',
//     },
//     {
//       name: 'Binapani Nursing Home and Hospital',
//       lat: 26.4905568,
//       lng: 90.96594469999999,
//       address: 'Rd, Kalahabhanga, Barpeta Road, Assam 781315, India',
//     },
//     {
//       name: 'Barpeta Road FRU',
//       lat: 26.4906224,
//       lng: 90.9685956,
//       address: 'FXR9+6CX, FRU, Main Road, Barpeta Road, Sat Bainer Tup, Assam 781315, India',
//     },
//     {
//       name: 'NIRC & Hospital',
//       lat: 26.3354151,
//       lng: 91.0019576,
//       address: '82P2+5Q8, Sonkuchi-Betbari Rd, Sonkuchi, Ranga Pani, Barpeta, Assam 781314, India',
//     },
//   ];

//   const isPointInPolygon = (point, polygon) => {
//     let inside = false;

//     for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
//       const xi = polygon[i][0];
//       const yi = polygon[i][1];
//       const xj = polygon[j][0];
//       const yj = polygon[j][1];

//       const intersect =
//         yi > point[1] !== yj > point[1] &&
//         point[0] < ((xj - xi) * (point[1] - yi)) / (yj - yi) + xi;

//       if (intersect) inside = !inside;
//     }

//     return inside;
//   };

//   return (
//     <div>
//       <svg width="300" height="300">
//         <polygon
//           points={polygon.map((point) => point.join(',')).join(' ')}
//           fill="lightgray"
//         />
//         {hospitals.map((hospital, index) => (
//           <circle
//             key={index}
//             cx={hospital.lng}
//             cy={hospital.lat}
//             r="3"
//             fill={isPointInPolygon([hospital.lng, hospital.lat], polygon) ? 'green' : 'red'}
//           />
//         ))}
//       </svg>
//       <div>
//         {hospitals.map((hospital, index) => (
//           <div key={index}>
//             {hospital.name}: ({hospital.lng}, {hospital.lat}) -{' '}
//             {isPointInPolygon([hospital.lng, hospital.lat], polygon) ? 'Inside' : 'Outside'}
//           </div>
//         ))}
//       </div>
//     </div>
//   );
// };

// export default PointInPolygon;

// import React, { useState, useEffect } from 'react';
// import firebase from 'firebase/app';
// import { app } from "../firebase-config";
// import 'firebase/firestore';

// const PointInPolygon = () => {
//   const [polygon, setPolygon] = useState([
//     [90.98953316168216, 26.323165319486552],
//     [91.02395126776126, 26.323165319486552],
//     [91.02395126776126, 26.342165634762075],
//     [90.98953316168216, 26.342165634762075],
//   ]);

//   const hospitals = [
//     {
//       name: 'Barpeta Civil Hospital, Kalgachia',
//       lat: 26.3556374,
//       lng: 90.8679944,
//       address: 'Civil Hospital, Road, Kalgachia, Assam 781319, India',
//     },
//     {
//       name: 'Sanjivani Hospital',
//       lat: 26.3252586,
//       lng: 90.9986036,
//       address: '8XGX+4C5, Barpeta-Hospital-Jania Rd, Muslimpatty, Barpeta, Assam 781301, India',
//     },
//     {
//       name: 'Binapani Nursing Home and Hospital',
//       lat: 26.4905568,
//       lng: 90.96594469999999,
//       address: 'Rd, Kalahabhanga, Barpeta Road, Assam 781315, India',
//     },
//     {
//       name: 'Barpeta Road FRU',
//       lat: 26.4906224,
//       lng: 90.9685956,
//       address: 'FXR9+6CX, FRU, Main Road, Barpeta Road, Sat Bainer Tup, Assam 781315, India',
//     },
//     {
//       name: 'NIRC & Hospital',
//       lat: 26.3354151,
//       lng: 91.0019576,
//       address: '82P2+5Q8, Sonkuchi-Betbari Rd, Sonkuchi, Ranga Pani, Barpeta, Assam 781314, India',
//     },
//   ];

//   useEffect(() => {
//     const db = app.firestore();
//     const alertsRef = db.collection('data_alerts');

//     const hospitalsInsidePolygon = hospitals.filter((hospital) =>
//       isPointInPolygon([hospital.lng, hospital.lat], polygon)
//     );

//     const dataToUpload = {
//       critical_infrastructure: hospitalsInsidePolygon,
//     };

//     // Add a new document with a generated ID
//     alertsRef
//       .add(dataToUpload)
//       .then((docRef) => {
//         console.log('Document written with ID: ', docRef.id);
//       })
//       .catch((error) => {
//         console.error('Error adding document: ', error);
//       });
//   }, [polygon]);

//   const isPointInPolygon = (point, polygon) => {
//     let inside = false;

//     for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
//       const xi = polygon[i][0];
//       const yi = polygon[i][1];
//       const xj = polygon[j][0];
//       const yj = polygon[j][1];

//       const intersect =
//         yi > point[1] !== yj > point[1] &&
//         point[0] < ((xj - xi) * (point[1] - yi)) / (yj - yi) + xi;

//       if (intersect) inside = !inside;
//     }

//     return inside;
//   };

//   return (
//     <div>
//       {/* Your existing SVG code here */}

//       <div>
//         {hospitals.map((hospital, index) => (
//           <div key={index}>
//             {hospital.name}: ({hospital.lng}, {hospital.lat}) -{' '}
//             {isPointInPolygon([hospital.lng, hospital.lat], polygon) ? 'Inside' : 'Outside'}
//           </div>
//         ))}
//       </div>
//     </div>
//   );
// };

// export default PointInPolygon;

// import React, { useState, useEffect } from "react";
// import { getFirestore, collection, addDoc } from "firebase/firestore";
// import { auth, db, storage } from "../firebase-config";

// const PointInPolygon = () => {
//   const [polygon, setPolygon] = useState([
//     [90.98953316168216, 26.323165319486552],
//     [91.02395126776126, 26.323165319486552],
//     [91.02395126776126, 26.342165634762075],
//     [90.98953316168216, 26.342165634762075],
//   ]);

//   const hospitals = [
//     {
//       name: "Barpeta Civil Hospital, Kalgachia",
//       lat: 26.3556374,
//       lng: 90.8679944,
//       address: "Civil Hospital, Road, Kalgachia, Assam 781319, India",
//     },
//     {
//       name: "Sanjivani Hospital",
//       lat: 26.3252586,
//       lng: 90.9986036,
//       address:
//         "8XGX+4C5, Barpeta-Hospital-Jania Rd, Muslimpatty, Barpeta, Assam 781301, India",
//     },
//     {
//       name: "Binapani Nursing Home and Hospital",
//       lat: 26.4905568,
//       lng: 90.96594469999999,
//       address: "Rd, Kalahabhanga, Barpeta Road, Assam 781315, India",
//     },
//     {
//       name: "Barpeta Road FRU",
//       lat: 26.4906224,
//       lng: 90.9685956,
//       address:
//         "FXR9+6CX, FRU, Main Road, Barpeta Road, Sat Bainer Tup, Assam 781315, India",
//     },
//     {
//       name: "NIRC & Hospital",
//       lat: 26.3354151,
//       lng: 91.0019576,
//       address:
//         "82P2+5Q8, Sonkuchi-Betbari Rd, Sonkuchi, Ranga Pani, Barpeta, Assam 781314, India",
//     },
//   ];

//   const [hospitalsInsidePolygon, setHospitalsInsidePolygon] = useState([]);

//   const isPointInPolygon = (point, polygon) => {
//     let inside = false;

//     for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
//       const xi = polygon[i][0];
//       const yi = polygon[i][1];
//       const xj = polygon[j][0];
//       const yj = polygon[j][1];

//       const intersect =
//         yi > point[1] !== yj > point[1] &&
//         point[0] < ((xj - xi) * (point[1] - yi)) / (yj - yi) + xi;

//       if (intersect) inside = !inside;
//     }

//     return inside;
//   };

//   useEffect(() => {
//     const insideHospitals = hospitals.filter((hospital) =>
//       isPointInPolygon([hospital.lng, hospital.lat], polygon)
//     );
//     setHospitalsInsidePolygon(insideHospitals);

//     // Upload to Firestore
//     const firestore = getFirestore(); // Get Firestore instance
//     insideHospitals.forEach(async (hospital) => {
//       try {
//         const docRef = await addDoc(collection(firestore, "hospitals"), {
//           name: hospital.name,
//           lat: hospital.lat,
//           lng: hospital.lng,
//           address: hospital.address,
//         });
//         console.log("Document written with ID: ", docRef.id);
//       } catch (e) {
//         console.error("Error adding document: ", e);
//       }
//     });
//   }, [polygon, hospitals]);

// //   const docRef = await addDoc(collection(db, "products"), {
// //     title: product.title,
// //     subtitle: product.subtitle,
// //     brand: product.brand,
// //     category: product.category,
// //     description: product.description,
// //     imageUrl,
// //     file1Url,
// //     file2Url,
// //   });

//   return (
//     <div>
//       {/* Your existing SVG code here */}

//       <div>
//         {hospitals.map((hospital, index) => (
//           <div key={index}>
//             {hospital.name}: ({hospital.lng}, {hospital.lat}) -{" "}
//             {isPointInPolygon([hospital.lng, hospital.lat], polygon)
//               ? "Inside"
//               : "Outside"}
//           </div>
//         ))}
//       </div>
//     </div>
//   );
// };

// export default PointInPolygon;

// import React, { useState, useEffect } from "react";
// import { getFirestore, collection, addDoc } from "firebase/firestore";
// import { auth, db, storage } from "../firebase-config";

// const PointInPolygon = () => {
//   const [polygon, setPolygon] = useState([
//     [90.98953316168216, 26.323165319486552],
//     [91.02395126776126, 26.323165319486552],
//     [91.02395126776126, 26.342165634762075],
//     [90.98953316168216, 26.342165634762075],
//   ]);

//   const hospitals = [
//     {
//       name: "Barpeta Civil Hospital, Kalgachia",
//       lat: 26.3556374,
//       lng: 90.8679944,
//       address: "Civil Hospital, Road, Kalgachia, Assam 781319, India",
//     },
//     {
//       name: "Sanjivani Hospital",
//       lat: 26.3252586,
//       lng: 90.9986036,
//       address:
//         "8XGX+4C5, Barpeta-Hospital-Jania Rd, Muslimpatty, Barpeta, Assam 781301, India",
//     },
//     {
//       name: "Binapani Nursing Home and Hospital",
//       lat: 26.4905568,
//       lng: 90.96594469999999,
//       address: "Rd, Kalahabhanga, Barpeta Road, Assam 781315, India",
//     },
//     {
//       name: "Barpeta Road FRU",
//       lat: 26.4906224,
//       lng: 90.9685956,
//       address:
//         "FXR9+6CX, FRU, Main Road, Barpeta Road, Sat Bainer Tup, Assam 781315, India",
//     },
//     {
//       name: "NIRC & Hospital",
//       lat: 26.3354151,
//       lng: 91.0019576,
//       address:
//         "82P2+5Q8, Sonkuchi-Betbari Rd, Sonkuchi, Ranga Pani, Barpeta, Assam 781314, India",
//     },
//   ];

//   const [hospitalsInsidePolygon, setHospitalsInsidePolygon] = useState([]);

//   const isPointInPolygon = (point, polygon) => {
//     let inside = false;

//     for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
//       const xi = polygon[i][0];
//       const yi = polygon[i][1];
//       const xj = polygon[j][0];
//       const yj = polygon[j][1];

//       const intersect =
//         yi > point[1] !== yj > point[1] &&
//         point[0] < ((xj - xi) * (point[1] - yi)) / (yj - yi) + xi;

//       if (intersect) inside = !inside;
//     }

//     return inside;
//   };

//   useEffect(() => {
//     // Fetch data without updating state or Firestore
//     const fetchData = async () => {
//       const insideHospitals = hospitals.filter((hospital) =>
//         isPointInPolygon([hospital.lng, hospital.lat], polygon)
//       );

//       // Upload to Firestore
//       const data = insideHospitals.map((hospital) => ({
//         name: hospital.name,
//         lat: hospital.lat,
//         lng: hospital.lng,
//         address: hospital.address,
//       }));

//       try {
//         const docRef = await addDoc(collection(db, "data_alerts"), {
//           critical_infrastructure: data,
//         });
//         console.log("Document written with ID: ", docRef.id);
//       } catch (e) {
//         console.error("Error adding document: ", e);
//       }
//     };

//     fetchData();
//   }, [polygon, hospitals]); // Run only when polygon or hospitals change

//   useEffect(() => {
//     // Update state after fetching data
//     const updateState = async () => {
//       const insideHospitals = hospitals.filter((hospital) =>
//         isPointInPolygon([hospital.lng, hospital.lat], polygon)
//       );
//       setHospitalsInsidePolygon(insideHospitals);
//     };

//     updateState();
//   }, [polygon, hospitals]); // Run only when polygon or hospitals change

//   return (
//     <div>
//       <div>
//         {hospitals.map((hospital, index) => (
//           <div key={index}>
//             {hospital.name}: ({hospital.lng}, {hospital.lat}) -{" "}
//             {isPointInPolygon([hospital.lng, hospital.lat], polygon)
//               ? "Inside"
//               : "Outside"}
//           </div>
//         ))}
//       </div>
//     </div>
//   );
// };

// export default PointInPolygon;

// This code is not uploading the data into the firebase firestore which lies in the polygon.

// ===============
// import React, { useState, useEffect } from "react";
// import { getFirestore, collection, addDoc } from "firebase/firestore"; // Use the compat version
// import { auth, db, storage } from "../firebase-config";

// const PointInPolygon = () => {
//   const [polygon, setPolygon] = useState([
//     [90.98953316168216, 26.323165319486552],
//     [91.02395126776126, 26.323165319486552],
//     [91.02395126776126, 26.342165634762075],
//     [90.98953316168216, 26.342165634762075],
//   ]);

//   const hospitals = [
//         {
//           name: "Barpeta Civil Hospital, Kalgachia",
//           lat: 26.3556374,
//           lng: 90.8679944,
//           address: "Civil Hospital, Road, Kalgachia, Assam 781319, India",
//         },
//         {
//           name: "Sanjivani Hospital",
//           lat: 26.3252586,
//           lng: 90.9986036,
//           address:
//             "8XGX+4C5, Barpeta-Hospital-Jania Rd, Muslimpatty, Barpeta, Assam 781301, India",
//         },
//         {
//           name: "Binapani Nursing Home and Hospital",
//           lat: 26.4905568,
//           lng: 90.96594469999999,
//           address: "Rd, Kalahabhanga, Barpeta Road, Assam 781315, India",
//         },
//         {
//           name: "Barpeta Road FRU",
//           lat: 26.4906224,
//           lng: 90.9685956,
//           address:
//             "FXR9+6CX, FRU, Main Road, Barpeta Road, Sat Bainer Tup, Assam 781315, India",
//         },
//         {
//           name: "NIRC & Hospital",
//           lat: 26.3354151,
//           lng: 91.0019576,
//           address:
//             "82P2+5Q8, Sonkuchi-Betbari Rd, Sonkuchi, Ranga Pani, Barpeta, Assam 781314, India",
//         },
//       ];

//   const [hospitalsInsidePolygon, setHospitalsInsidePolygon] = useState([]);

//   const isPointInPolygon = (point, polygon) => {
//     let inside = false;

//     for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
//       const xi = polygon[i][0];
//       const yi = polygon[i][1];
//       const xj = polygon[j][0];
//       const yj = polygon[j][1];

//       const intersect =
//         yi > point[1] !== yj > point[1] &&
//         point[0] < ((xj - xi) * (point[1] - yi)) / (yj - yi) + xi;

//       if (intersect) inside = !inside;
//     }

//     return inside;
//   };

//   useEffect(() => {
//     // Fetch data without updating state or Firestore
//     const fetchData = async () => {
//       const insideHospitals = hospitals.filter((hospital) =>
//         isPointInPolygon([hospital.lng, hospital.lat], polygon)
//       );

//       // Upload to Firestore
//       const data = insideHospitals.map((hospital) => ({
//         name: hospital.name,
//         lat: hospital.lat,
//         lng: hospital.lng,
//         address: hospital.address,
//       }));

//       try {
//         const docRef = await addDoc(collection(db, "data_alerts"), {
//           critical_infrastructure: data,
//         });
//         console.log("Document written with ID: ", docRef.id);
//       } catch (e) {
//         console.error("Error adding document: ", e);
//       }
//     };

//     fetchData();
//   }, [polygon, hospitals]); // Run only when polygon or hospitals change

//   useEffect(() => {
//     // Update state after fetching data
//     const updateState = async () => {
//       const insideHospitals = hospitals.filter((hospital) =>
//         isPointInPolygon([hospital.lng, hospital.lat], polygon)
//       );
//       setHospitalsInsidePolygon(insideHospitals);
//     };

//     updateState();
//   }, [polygon, hospitals]); // Run only when polygon or hospitals change

//   return (
//     <div>
//       <div>
//         {hospitals.map((hospital, index) => (
//           <div key={index}>
//             {hospital.name}: ({hospital.lng}, {hospital.lat}) -{" "}
//             {isPointInPolygon([hospital.lng, hospital.lat], polygon)
//               ? "Inside"
//               : "Outside"}
//           </div>
//         ))}
//       </div>
//     </div>
//   );
// };

// export default PointInPolygon;
// ==============================

// import React, { useState, useEffect, useMemo } from "react";
// import { getFirestore, collection, addDoc } from "firebase/firestore"; // Use the compat version
// import { auth, db, storage } from "../firebase-config";

// const PointInPolygon = () => {
//   const [polygon, setPolygon] = useState([
//     [90.98953316168216, 26.323165319486552],
//     [91.02395126776126, 26.323165319486552],
//     [91.02395126776126, 26.342165634762075],
//     [90.98953316168216, 26.342165634762075],
//   ]);

//   const hospitals = useMemo(
//     () => [
//       {
//         name: "Barpeta Civil Hospital, Kalgachia",
//         lat: 26.3556374,
//         lng: 90.8679944,
//         address: "Civil Hospital, Road, Kalgachia, Assam 781319, India",
//       },
//       {
//         name: "Sanjivani Hospital",
//         lat: 26.3252586,
//         lng: 90.9986036,
//         address:
//           "8XGX+4C5, Barpeta-Hospital-Jania Rd, Muslimpatty, Barpeta, Assam 781301, India",
//       },
//       {
//         name: "Binapani Nursing Home and Hospital",
//         lat: 26.4905568,
//         lng: 90.96594469999999,
//         address: "Rd, Kalahabhanga, Barpeta Road, Assam 781315, India",
//       },
//       {
//         name: "Barpeta Road FRU",
//         lat: 26.4906224,
//         lng: 90.9685956,
//         address:
//           "FXR9+6CX, FRU, Main Road, Barpeta Road, Sat Bainer Tup, Assam 781315, India",
//       },
//       {
//         name: "NIRC & Hospital",
//         lat: 26.3354151,
//         lng: 91.0019576,
//         address:
//           "82P2+5Q8, Sonkuchi-Betbari Rd, Sonkuchi, Ranga Pani, Barpeta, Assam 781314, India",
//       },
//     ],
//     [
//       /* Dependencies can be added here if needed */
//     ]
//   );

//   const [hospitalsInsidePolygon, setHospitalsInsidePolygon] = useState([]);

//   const isPointInPolygon = (point, polygon) => {
//     let inside = false;

//     for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
//       const xi = polygon[i][0];
//       const yi = polygon[i][1];
//       const xj = polygon[j][0];
//       const yj = polygon[j][1];

//       const intersect =
//         yi > point[1] !== yj > point[1] &&
//         point[0] < ((xj - xi) * (point[1] - yi)) / (yj - yi) + xi;

//       if (intersect) inside = !inside;
//     }

//     return inside;
//   };

//   useEffect(() => {
//     // Fetch data without updating state or Firestore
//     const fetchData = async () => {
//       const insideHospitals = hospitals.filter((hospital) =>
//         isPointInPolygon([hospital.lng, hospital.lat], polygon)
//       );

//       // Upload to Firestore
//       const data = insideHospitals.map((hospital) => ({
//         name: hospital.name,
//         lat: hospital.lat,
//         lng: hospital.lng,
//         address: hospital.address,
//       }));

//       try {
//         const docRef = await addDoc(collection(db, "data_alerts"), {
//           critical_infrastructure: data,
//         });
//         console.log("Document written with ID: ", docRef.id);
//       } catch (e) {
//         console.error("Error adding document: ", e);
//       }
//     };

//     fetchData();
//   }, [polygon, hospitals]); // Run only when polygon or hospitals change

//   useEffect(() => {
//     // Update state after fetching data
//     const updateState = async () => {
//       const insideHospitals = hospitals.filter((hospital) =>
//         isPointInPolygon([hospital.lng, hospital.lat], polygon)
//       );
//       setHospitalsInsidePolygon(insideHospitals);
//     };

//     updateState();
//   }, [polygon, hospitals]); // Run only when polygon or hospitals change

//   return (
//     <div>
//       <div>
//         {hospitals.map((hospital, index) => (
//           <div key={index}>
//             {hospital.name}: ({hospital.lng}, {hospital.lat}) -{" "}
//             {isPointInPolygon([hospital.lng, hospital.lat], polygon)
//               ? "Inside"
//               : "Outside"}
//           </div>
//         ))}
//       </div>
//     </div>
//   );
// };

// export default PointInPolygon;


























// fixed infinite loop

import React, { useState, useEffect, useMemo } from "react";
import { getFirestore, collection, addDoc } from "firebase/firestore";
import { auth, db, storage } from "../firebase-config";

const PointInPolygon = () => {
  const [polygon, setPolygon] = useState([
    [90.98953316168216, 26.323165319486552],
    [91.02395126776126, 26.323165319486552],
    [91.02395126776126, 26.342165634762075],
    [90.98953316168216, 26.342165634762075],
  ]);

  const hospitals = useMemo(
        () => [
          {
            name: "Barpeta Civil Hospital, Kalgachia",
            lat: 26.3556374,
            lng: 90.8679944,
            address: "Civil Hospital, Road, Kalgachia, Assam 781319, India",
          },
          {
            name: "Sanjivani Hospital",
            lat: 26.3252586,
            lng: 90.9986036,
            address:
              "8XGX+4C5, Barpeta-Hospital-Jania Rd, Muslimpatty, Barpeta, Assam 781301, India",
          },
          {
            name: "Binapani Nursing Home and Hospital",
            lat: 26.4905568,
            lng: 90.96594469999999,
            address: "Rd, Kalahabhanga, Barpeta Road, Assam 781315, India",
          },
          {
            name: "Barpeta Road FRU",
            lat: 26.4906224,
            lng: 90.9685956,
            address:
              "FXR9+6CX, FRU, Main Road, Barpeta Road, Sat Bainer Tup, Assam 781315, India",
          },
          {
            name: "NIRC & Hospital",
            lat: 26.3354151,
            lng: 91.0019576,
            address:
              "82P2+5Q8, Sonkuchi-Betbari Rd, Sonkuchi, Ranga Pani, Barpeta, Assam 781314, India",
          },
        ],
        [
          /* Dependencies can be added here if needed */
        ]
      );

  const [hospitalsInsidePolygon, setHospitalsInsidePolygon] = useState([]);

  const isPointInPolygon = (point, polygon) => {
    let inside = false;

    for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
      const xi = polygon[i][0];
      const yi = polygon[i][1];
      const xj = polygon[j][0];
      const yj = polygon[j][1];

      const intersect =
        yi > point[1] !== yj > point[1] &&
        point[0] < ((xj - xi) * (point[1] - yi)) / (yj - yi) + xi;

      if (intersect) inside = !inside;
    }

    return inside;
  };

  useEffect(() => {
    // Fetch data without updating state or Firestore
    const fetchData = async () => {
      const insideHospitals = hospitals.filter((hospital) =>
        isPointInPolygon([hospital.lng, hospital.lat], polygon)
      );

      // Upload to Firestore
      const data = insideHospitals.map((hospital) => ({
        name: hospital.name,
        lat: hospital.lat,
        lng: hospital.lng,
        address: hospital.address,
      }));

      try {
        const docRef = await addDoc(collection(db, "data_alerts"), {
          critical_infrastructure: data,
        });
        console.log("Document written with ID: ", docRef.id);
      } catch (e) {
        console.error("Error adding document: ", e);
      }
    };

    fetchData();
  }, [polygon, hospitals]); // Run only when polygon or hospitals change

  useEffect(() => {
    // Update state after fetching data
    const updateState = async () => {
      const insideHospitals = hospitals.filter((hospital) =>
        isPointInPolygon([hospital.lng, hospital.lat], polygon)
      );
      setHospitalsInsidePolygon(insideHospitals);
    };

    updateState();
  }, [polygon, hospitals]); // Run only when polygon or hospitals change

  return (
    <div>
      <div>
        {hospitals.map((hospital, index) => (
          <div key={index}>
            {hospital.name}: ({hospital.lng}, {hospital.lat}) -{" "}
            {isPointInPolygon([hospital.lng, hospital.lat], polygon)
              ? "Inside"
              : "Outside"}
          </div>
        ))}
      </div>
    </div>
  );
};

export default PointInPolygon;
