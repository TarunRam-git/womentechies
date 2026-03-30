import { Server } from "socket.io";

export const config = {
  api: {
    bodyParser: false
  }
};

let io;

export default function handler(req, res) {
  if (!res.socket.server.io) {
    io = new Server(res.socket.server, {
      path: "/api/socketio",
      addTrailingSlash: false
    });
    res.socket.server.io = io;

    io.on("connection", (socket) => {
      socket.on("join-room", ({ room, name, role }) => {
        if (!room) return;
        socket.data.name = name;
        socket.data.role = role;
        socket.join(room);
        socket.to(room).emit("system", { text: `${name || "Partner"} joined the room` });
      });

      socket.on("send-message", ({ room, text, name }) => {
        if (!room || !text) return;
        const payload = { from: name || "Anon", text, ts: Date.now() };
        io.to(room).emit("new-message", payload);
      });
    });
  }
  res.end();
}
