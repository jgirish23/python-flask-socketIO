const chatbotToggler = document.querySelector(".chatbot-toggler");
const closeBtn = document.querySelector(".close-btn");
const chatbox = document.querySelector(".chatbox");
const chatInput = document.querySelector(".chat-input textarea");
const sendChatBtn = document.querySelector(".chat-input span");

$(document).ready(function() {
    $('.chat-input').submit(function(event) {
        event.preventDefault(); // Prevent the default form submission

        var userMessage = $('#user-message').val();

        $.ajax({
            type: 'POST',
            url: '/',
            contentType: 'application/json', // Set content type to JSON
            data: JSON.stringify({user_message: userMessage}),
            success: function(response) {
                // Handle the response from Flask
                console.log(response);
            },
            error: function(xhr, status, error) {
                console.error(error);
            }
        });

        handleChat();
    });
});

// Connect to the WebSocket server
const socket = io.connect('http://localhost:5000');

// Handle incoming messages from the server
// socket.on('response', (data) => {
//     console.log('Received response:', data);
//     // Handle the response data
// });

// // Send a message to the server
// socket.emit('message', { someData: 'Hello, server!' });


const inputInitHeight = chatInput.scrollHeight;

const createChatLi = (message, className) => {
    // Create a chat <li> element with passed message and className
    const chatLi = document.createElement("li");
    chatLi.classList.add("chat", `${className}`);
    let chatContent = className === "outgoing" ? `<p></p>` : `<span class="material-symbols-outlined">smart_toy</span><p></p>`;
    chatLi.innerHTML = chatContent;
    chatLi.querySelector("p").textContent = message;
    return chatLi; // return chat <li> element
}

// const generateResponse = (chatElement) => {
//     const messageElement = chatElement.querySelector("p");
//     console.log("msg",messageElement);

//     socket.on('server_message', (data) => {
//         console.log('Received response:', data);
//         // Handle the response data
//         // const incomingChatLi = createChatLi(data["message"]);
//         // chatbox.appendChild(incomingChatLi);
//         messageElement.textContent = data["message"];

//     });
//     socket.emit('message_n', { someData_n: userMessage });
//     console.log("Done!")
// }

const handleChat = () => {
    userMessage = chatInput.value.trim(); // Get user entered message and remove extra whitespace
    console.log(userMessage);
    if(!userMessage) return;
    // Send a message to the server
    socket.emit('message', { someData: userMessage });

    // Clear the input textarea and set its height to default
    chatInput.value = "";
    chatInput.style.height = `${inputInitHeight}px`;

    // Append the user's message to the chatbox
    chatbox.appendChild(createChatLi(userMessage, "outgoing"));
    chatbox.scrollTo(0, chatbox.scrollHeight);

}

socket.on('server_message', (data) => {
    console.log('Received response:', data);
    // Handle the response data

    const incomingChatLi = createChatLi(data["message"], "incoming");
    chatbox.appendChild(incomingChatLi);
    chatbox.scrollTo(0, chatbox.scrollHeight);
    // messageElement.textContent = data["message"];
});


chatInput.addEventListener("input", () => {
    // Adjust the height of the input textarea based on its content
    chatInput.style.height = `${inputInitHeight}px`;
    chatInput.style.height = `${chatInput.scrollHeight}px`;
});

sendChatBtn.addEventListener("click", handleChat);
closeBtn.addEventListener("click", () => {document.body.classList.remove("show-chatbot");});
chatbotToggler.addEventListener("click", () => {document.body.classList.toggle("show-chatbot");});