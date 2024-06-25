// //  For connecting DB with node js
// const mongoose = require("mongoose");

// mongoose.connect("mongodb://localhost:27017/", {
    
// }).then(()=>{
//     console.log('Connection Successful');
// }).catch((err)=>{
//     console.log(err);
// });
const mongoose = require("mongoose");

const connectDB = async () => {
    try {
        await mongoose.connect("mongodb://127.0.0.1:27017/ImageCap", {
            useNewUrlParser: true,
            useUnifiedTopology: true,
        });
        console.log('Connection Successful');
    } catch (err) {
        console.error('Connection error', err);
        // Optionally, you can retry the connection after some time
        setTimeout(connectDB, 5000); // Retry connection after 5 seconds
    }
};

connectDB();
