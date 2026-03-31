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
      addTrailingSlash: false,
      cors: {
        origin: "*",
        methods: ["GET", "POST"],
      },
    });
    res.socket.server.io = io;

    io.on("connection", (socket) => {
      const emitRoomPeers = (room) => {
        if (!room) return;
        const peers = [];
        for (const s of io.sockets.adapter.rooms.get(room) || []) {
          const client = io.sockets.sockets.get(s);
          if (!client) continue;
          peers.push({ id: client.id, name: client.data.name || "Partner", role: client.data.role || "user" });
        }
        io.to(room).emit("room-peers", peers);
      };

      socket.on("join-room", ({ room, name, role }) => {
        if (!room) return;
        const prevRoom = socket.data.room;
        if (prevRoom && prevRoom !== room) {
          socket.leave(prevRoom);
          emitRoomPeers(prevRoom);
        }

        socket.data.name = name;
        socket.data.role = role;
        socket.data.room = room;
        socket.join(room);
        socket.to(room).emit("system", { text: `${name || "Partner"} joined the room` });
        emitRoomPeers(room);
      });

      socket.on("send-message", ({ room, text, name }) => {
        const targetRoom = room || socket.data.room;
        const clean = String(text || "").trim();
        if (!targetRoom || !clean) return;
        const payload = { from: name || socket.data.name || "Anon", text: clean, ts: Date.now() };
        io.to(targetRoom).emit("new-message", payload);
      });

      socket.on("avatar-motion", ({ room, motion }) => {
        const targetRoom = room || socket.data.room;
        if (!targetRoom || !motion) return;
        socket.to(targetRoom).emit("avatar-motion", {
          from: socket.data.name || "Partner",
          motion,
          ts: Date.now(),
        });
      });

      socket.on("video-frame", ({ room, frame }) => {
        const targetRoom = room || socket.data.room;
        if (!targetRoom || !frame) return;
        socket.to(targetRoom).emit("video-frame", {
          from: socket.data.name || "Partner",
          frame,
          ts: Date.now(),
        });
      });

      socket.on("disconnect", () => {
        const room = socket.data.room;
        if (!room) return;
        socket.to(room).emit("system", { text: `${socket.data.name || "Partner"} left the room` });
        emitRoomPeers(room);
      });
    });
  }
  res.end();
}
