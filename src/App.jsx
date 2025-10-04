import React, { useState, useEffect, useRef } from 'react';

// Firebase imports
import { initializeApp } from "firebase/app";
import {
    getAuth,
    onAuthStateChanged,
    signOut,
    sendPasswordResetEmail,
    createUserWithEmailAndPassword,
    signInWithEmailAndPassword,
    fetchSignInMethodsForEmail,
    confirmPasswordReset,
    verifyPasswordResetCode
} from "firebase/auth";
import { getFirestore, collection, addDoc, query, where, onSnapshot, doc, deleteDoc, serverTimestamp, orderBy, updateDoc, getDoc, setDoc } from "firebase/firestore";
import { getStorage, ref, uploadBytesResumable, getDownloadURL, deleteObject } from "firebase/storage";

// --- START OF FIREBASE CONFIG ---
const firebaseConfig = {
    apiKey: process.env.REACT_APP_FIREBASE_API_KEY,
    authDomain: process.env.REACT_APP_FIREBASE_AUTH_DOMAIN,
    projectId: process.env.REACT_APP_FIREBASE_PROJECT_ID,
    storageBucket: process.env.REACT_APP_FIREBASE_STORAGE_BUCKET,
    messagingSenderId: process.env.REACT_APP_FIREBASE_MESSAGING_SENDER_ID,
    appId: process.env.REACT_APP_FIREBASE_APP_ID
};

const app = initializeApp(firebaseConfig);
const auth = getAuth(app);
const db = getFirestore(app);
const storage = getStorage(app);
// --- END OF FIREBASE CONFIG ---

// --- SVG Icons ---
const DashboardIcon = () => <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h7" /></svg>;
const ChatIcon = () => <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" /></svg>;
const UploadIcon = () => <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" /></svg>;
const DeleteIcon = () => <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" /></svg>;
const FilePdfIcon = () => <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" /></svg>;
const SendIcon = () => <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" /></svg>;
const LogoutIcon = () => <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H7a3 3 0 01-3-3v-10a3 3 0 013-3h3a3 3 0 013 3v1" /></svg>;
const ArrowLeftIcon = () => <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" /></svg>;
const PlusIcon = () => <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" /></svg>;
const MenuIcon = () => <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" /></svg>;
const XIcon = () => <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" /></svg>;
const EyeIcon = () => <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0zm6 0c0 5.523-7.462 9-9 9s-9-3.477-9-9 7.462-9 9-9 9 3.477 9 9z" /></svg>;
const EyeOffIcon = () => <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.875 18.825A10.05 10.05 0 0112 19c-4.477 0-8.268-2.943-9.542-7a9.977 9.977 0 012.133-3.825m3.91-.825A7.002 7.002 0 0112 5c1.725 0 3.325.575 4.625 1.525M3 3l18 18" /></svg>;

// --- RAG ChatBot Component ---
function ChatBot({ user, onBack }) {
    const [messages, setMessages] = useState([]);
    const [inputMessage, setInputMessage] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [ragStatus, setRagStatus] = useState('loading');
    const [chatSessions, setChatSessions] = useState([]);
    const [currentChatId, setCurrentChatId] = useState(null);
    const [sidebarOpen, setSidebarOpen] = useState(false);
    const messagesEndRef = useRef(null);
    const chatContainerRef = useRef(null);

    const API_BASE_URL = process.env.REACT_APP_RAG_API_URL || 'http://localhost:8000';

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    useEffect(() => {
        checkRagStatus();
        loadChatSessions();
    }, []);

    useEffect(() => {
        if (currentChatId) {
            loadChatMessages(currentChatId);
        }
    }, [currentChatId]);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    const checkRagStatus = async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/status`);
            if (response.ok) {
                const data = await response.json();
                setRagStatus(data.status);
            } else {
                setRagStatus('error');
            }
        } catch (error) {
            console.error('Error checking RAG status:', error);
            setRagStatus('error');
        }
    };

    const loadChatSessions = () => {
        const q = query(
            collection(db, "chatSessions"), 
            where("userId", "==", user.uid),
            orderBy("updatedAt", "desc")
        );
        
        onSnapshot(q, (querySnapshot) => {
            const sessions = querySnapshot.docs.map(doc => ({ 
                id: doc.id, 
                ...doc.data() 
            }));
            setChatSessions(sessions);
            
            // Si no hay chat actual y hay sesiones, cargar la más reciente
            if (!currentChatId && sessions.length > 0) {
                setCurrentChatId(sessions[0].id);
            }
        });
    };

    const loadChatMessages = async (chatId) => {
        try {
            const chatDoc = await getDoc(doc(db, "chatSessions", chatId));
            if (chatDoc.exists()) {
                const chatData = chatDoc.data();
                setMessages(chatData.messages || []);
            }
        } catch (error) {
            console.error('Error loading chat messages:', error);
        }
    };

    const createNewChat = async () => {
        const welcomeMessage = {
            id: 1,
            text: "Bienvenido. Soy tu asistente legal virtual. Mis respuestas se basan estrictamente en el Código del Trabajo de Honduras.\n\n**¿En qué puedo ayudarte hoy?**",
            isBot: true,
            timestamp: new Date()
        };

        try {
            const newChatRef = await addDoc(collection(db, "chatSessions"), {
                userId: user.uid,
                title: "Nueva conversación",
                messages: [welcomeMessage],
                createdAt: serverTimestamp(),
                updatedAt: serverTimestamp()
            });

            setCurrentChatId(newChatRef.id);
            setMessages([welcomeMessage]);
            setSidebarOpen(false);
        } catch (error) {
            console.error('Error creating new chat:', error);
        }
    };

    const saveChatMessage = async (newMessages, chatTitle = null) => {
        if (!currentChatId) return;

        try {
            const updateData = {
                messages: newMessages,
                updatedAt: serverTimestamp()
            };

            if (chatTitle) {
                updateData.title = chatTitle;
            }

            await updateDoc(doc(db, "chatSessions", currentChatId), updateData);
        } catch (error) {
            console.error('Error saving chat message:', error);
        }
    };

    const generateChatTitle = (message) => {
        // Generar un título basado en la primera pregunta del usuario
        const words = message.split(' ').slice(0, 6).join(' ');
        return words.length > 30 ? words.substring(0, 27) + '...' : words;
    };

    const sendMessage = async () => {
        if (!inputMessage.trim() || isLoading || ragStatus !== 'ready') return;

        const userMessage = {
            id: Date.now(),
            text: inputMessage.trim(),
            isBot: false,
            timestamp: new Date()
        };

        // Si no hay chat actual, crear uno nuevo
        if (!currentChatId) {
            await createNewChat();
        }

        const newMessages = [...messages, userMessage];
        setMessages(newMessages);
        
        // Si es el primer mensaje del usuario en este chat, actualizar el título
        const isFirstUserMessage = messages.length === 1; // Solo mensaje de bienvenida
        
        setInputMessage('');
        setIsLoading(true);

        // Preparar historial para la API
        const historyForAPI = messages.map(msg => ({
            role: msg.isBot ? 'assistant' : 'user',
            content: msg.text
        }));

        try {
            const response = await fetch(`${API_BASE_URL}/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: userMessage.text,
                    chat_history: historyForAPI,
                    user_id: user.uid,
                    chat_id: currentChatId
                })
            });

            if (response.ok) {
                const data = await response.json();
                const botMessage = {
                    id: Date.now() + 1,
                    text: data.response,
                    isBot: true,
                    timestamp: new Date(),
                    sources: data.sources || null
                };
                
                const finalMessages = [...newMessages, botMessage];
                setMessages(finalMessages);
                
                // Guardar mensajes en Firebase
                const chatTitle = isFirstUserMessage ? generateChatTitle(userMessage.text) : null;
                await saveChatMessage(finalMessages, chatTitle);
                
            } else {
                throw new Error('Error en la respuesta del servidor');
            }
        } catch (error) {
            console.error('Error sending message:', error);
            const errorMessage = {
                id: Date.now() + 1,
                text: "Lo siento, ha ocurrido un error. Por favor, inténtalo de nuevo más tarde.",
                isBot: true,
                timestamp: new Date(),
                isError: true
            };
            const finalMessages = [...newMessages, errorMessage];
            setMessages(finalMessages);
            await saveChatMessage(finalMessages);
        } finally {
            setIsLoading(false);
        }
    };

    const deleteChat = async (chatId) => {
        if (window.confirm("¿Estás seguro de que quieres eliminar esta conversación?")) {
            try {
                await deleteDoc(doc(db, "chatSessions", chatId));
                if (currentChatId === chatId) {
                    setCurrentChatId(null);
                    setMessages([]);
                }
            } catch (error) {
                console.error('Error deleting chat:', error);
            }
        }
    };

    const handleKeyPress = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    };

    const formatMessage = (text) => {
        return text
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/\n/g, '<br/>');
    };

    const getExampleQuestions = () => [
        "¿Cuál es la jornada máxima de trabajo diurno?",
        "¿Cuántos días de vacaciones corresponden después de 3 años de servicio?",
        "¿Qué dice la ley sobre el despido de una mujer embarazada?"
    ];

    const handleExampleClick = (question) => {
        setInputMessage(question);
    };

    return (
        <div className="flex h-screen bg-gray-900 text-white">
            {/* Sidebar */}
            <div className={`${sidebarOpen ? 'translate-x-0' : '-translate-x-full'} fixed inset-y-0 left-0 z-50 w-64 bg-gray-800 transform transition-transform duration-300 ease-in-out lg:translate-x-0 lg:static lg:inset-0`}>
                <div className="flex items-center justify-between p-4 border-b border-gray-700">
                    <h2 className="text-lg font-semibold">Conversaciones</h2>
                    <button
                        onClick={() => setSidebarOpen(false)}
                        className="lg:hidden p-2 rounded-md hover:bg-gray-700"
                    >
                        <XIcon />
                    </button>
                </div>
                
                <div className="p-4">
                    <button
                        onClick={createNewChat}
                        className="w-full flex items-center justify-center px-4 py-2 bg-indigo-600 hover:bg-indigo-700 rounded-md transition-colors"
                    >
                        <PlusIcon />
                        <span className="ml-2">Nueva conversación</span>
                    </button>
                </div>

                <div className="flex-1 overflow-y-auto">
                    {chatSessions.map((session) => (
                        <div
                            key={session.id}
                            className={`mx-2 mb-2 p-3 rounded-lg cursor-pointer transition-colors ${
                                currentChatId === session.id 
                                    ? 'bg-indigo-600' 
                                    : 'bg-gray-700 hover:bg-gray-600'
                            }`}
                            onClick={() => {
                                setCurrentChatId(session.id);
                                setSidebarOpen(false);
                            }}
                        >
                            <div className="flex items-center justify-between">
                                <div className="flex-1 min-w-0">
                                    <p className="text-sm font-medium truncate">
                                        {session.title || 'Nueva conversación'}
                                    </p>
                                    <p className="text-xs text-gray-400">
                                        {session.updatedAt?.toDate ? 
                                            session.updatedAt.toDate().toLocaleDateString() :
                                            'Reciente'
                                        }
                                    </p>
                                </div>
                                <button
                                    onClick={(e) => {
                                        e.stopPropagation();
                                        deleteChat(session.id);
                                    }}
                                    className="p-1 text-gray-400 hover:text-red-400 transition-colors"
                                >
                                    <DeleteIcon />
                                </button>
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            {/* Main Chat Area */}
            <div className="flex flex-col flex-1">
                {/* Header */}
                <header className="flex items-center justify-between p-4 bg-gray-800 border-b border-gray-700">
                    <div className="flex items-center">
                        <button
                            onClick={() => setSidebarOpen(true)}
                            className="lg:hidden p-2 mr-2 text-gray-300 hover:text-white transition-colors"
                        >
                            <MenuIcon />
                        </button>
                        <button
                            onClick={onBack}
                            className="flex items-center px-3 py-2 mr-4 text-gray-300 hover:text-white transition-colors"
                        >
                            <ArrowLeftIcon />
                        </button>
                        <ChatIcon />
                        <h1 className="ml-2 text-xl font-bold">Asistente Legal Virtual</h1>
                    </div>
                    <div className="flex items-center space-x-4">
                        <div className={`flex items-center px-3 py-1 rounded-full text-sm ${
                            ragStatus === 'ready' ? 'bg-green-900 text-green-300' :
                            ragStatus === 'loading' ? 'bg-yellow-900 text-yellow-300' :
                            'bg-red-900 text-red-300'
                        }`}>
                            {ragStatus === 'ready' ? '● Conectado' :
                             ragStatus === 'loading' ? '● Cargando...' :
                             '● Desconectado'}
                        </div>
                    </div>
                </header>

                {/* Chat Messages */}
                <div
                    ref={chatContainerRef}
                    className="flex-1 overflow-y-auto p-4 space-y-4"
                >
                    {ragStatus !== 'ready' && (
                        <div className="text-center p-8">
                            {ragStatus === 'loading' && (
                                <div className="text-yellow-400">
                                    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-yellow-400 mx-auto mb-4"></div>
                                    <p>Inicializando el asistente legal...</p>
                                    <p className="text-sm text-gray-400 mt-2">Esto puede tomar unos momentos</p>
                                </div>
                            )}
                            {ragStatus === 'error' && (
                                <div className="text-red-400">
                                    <p className="text-lg mb-2">Error de conexión</p>
                                    <p className="text-sm text-gray-400 mb-4">No se puede conectar con el servidor del asistente</p>
                                    <button
                                        onClick={checkRagStatus}
                                        className="px-4 py-2 bg-red-600 hover:bg-red-700 rounded-md transition-colors"
                                    >
                                        Reintentar
                                    </button>
                                </div>
                            )}
                        </div>
                    )}

                    {ragStatus === 'ready' && (
                        <>
                            {messages.map((message) => (
                                <div key={message.id} className={`flex ${message.isBot ? 'justify-start' : 'justify-end'}`}>
                                    <div className={`max-w-3xl p-4 rounded-lg ${
                                        message.isBot
                                            ? message.isError
                                                ? 'bg-red-900 border border-red-700'
                                                : 'bg-gray-800 border border-gray-700'
                                            : 'bg-indigo-600'
                                    }`}>
                                        <div
                                            className="whitespace-pre-wrap"
                                            dangerouslySetInnerHTML={{
                                                __html: formatMessage(message.text)
                                            }}
                                        />
                                        {message.sources && (
                                            <div className="mt-4 pt-3 border-t border-gray-600">
                                                <p className="text-sm font-semibold text-gray-300 mb-2">Fuentes consultadas:</p>
                                                <div className="text-xs text-gray-400 bg-gray-900 p-3 rounded max-h-32 overflow-y-auto">
                                                    {message.sources}
                                                </div>
                                            </div>
                                        )}
                                        <div className="text-xs text-gray-400 mt-2">
                                            {message.timestamp instanceof Date ? 
                                                message.timestamp.toLocaleTimeString() :
                                                new Date(message.timestamp).toLocaleTimeString()
                                            }
                                        </div>
                                    </div>
                                </div>
                            ))}

                            {isLoading && (
                                <div className="flex justify-start">
                                    <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
                                        <div className="flex items-center space-x-2">
                                            <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-indigo-400"></div>
                                            <span className="text-gray-400">El asistente está escribiendo...</span>
                                        </div>
                                    </div>
                                </div>
                            )}

                            {messages.length === 1 && (
                                <div className="mt-8">
                                    <h3 className="text-lg font-semibold text-gray-300 mb-4">Preguntas de ejemplo:</h3>
                                    <div className="space-y-2">
                                        {getExampleQuestions().map((question, index) => (
                                            <button
                                                key={index}
                                                onClick={() => handleExampleClick(question)}
                                                className="block w-full text-left p-3 bg-gray-800 hover:bg-gray-700 rounded-lg transition-colors text-sm border border-gray-600"
                                                disabled={isLoading}
                                            >
                                                {question}
                                            </button>
                                        ))}
                                    </div>
                                </div>
                            )}

                            <div ref={messagesEndRef} />
                        </>
                    )}
                </div>

                {/* Input Area */}
                {ragStatus === 'ready' && (
                    <div className="border-t border-gray-700 p-4 bg-gray-800">
                        <div className="flex space-x-4">
                            <div className="flex-1">
                                <textarea
                                    value={inputMessage}
                                    onChange={(e) => setInputMessage(e.target.value)}
                                    onKeyPress={handleKeyPress}
                                    placeholder="Escribe tu consulta aquí..."
                                    className="w-full p-3 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-400 resize-none focus:outline-none focus:ring-2 focus:ring-indigo-500"
                                    rows={1}
                                    style={{ minHeight: '44px', maxHeight: '120px' }}
                                    disabled={isLoading}
                                />
                            </div>
                            <button
                                onClick={sendMessage}
                                disabled={!inputMessage.trim() || isLoading}
                                className="px-6 py-3 bg-indigo-600 hover:bg-indigo-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded-lg font-semibold flex items-center space-x-2 transition-colors"
                            >
                                <SendIcon />
                                <span>Enviar</span>
                            </button>
                        </div>
                        <div className="mt-2 text-xs text-gray-400 text-center">
                            Las respuestas son generadas por IA y deben ser utilizadas como guía informativa, no como asesoría legal definitiva.
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}

// --- Admin Panel View ---
function Dashboard({ user }) {
    const [file, setFile] = useState(null);
    const [uploadProgress, setUploadProgress] = useState(0);
    const [filesList, setFilesList] = useState([]);
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(true);
    const [currentView, setCurrentView] = useState('dashboard');

    useEffect(() => {
        setLoading(true);
        const q = query(collection(db, "files"), where("userId", "==", user.uid), orderBy("createdAt", "desc"));
        const unsubscribe = onSnapshot(q, (querySnapshot) => {
            const filesData = querySnapshot.docs.map(doc => ({ id: doc.id, ...doc.data() }));
            setFilesList(filesData);
            setLoading(false);
        });
        return unsubscribe;
    }, [user.uid]);

    const handleFileChange = (e) => {
        const selectedFile = e.target.files[0];
        if (selectedFile) {
            if (selectedFile.type !== "application/pdf") {
                setError("Error: Solo se permiten archivos PDF.");
                return;
            }
            if (selectedFile.size > 5 * 1024 * 1024) {
                setError("Error: El archivo no debe exceder los 5MB.");
                return;
            }
            setFile(selectedFile);
            setError('');
        }
    };

    const handleUpload = () => {
        if (!file) return;

        const storageRef = ref(storage, `uploads/${user.uid}/${Date.now()}-${file.name}`);
        const uploadTask = uploadBytesResumable(storageRef, file);

        uploadTask.on('state_changed',
            (snapshot) => {
                const progress = (snapshot.bytesTransferred / snapshot.totalBytes) * 100;
                setUploadProgress(progress);
            },
            (error) => {
                console.error("Error al subir el archivo:", error);
                setError("Error al subir el archivo.");
            },
            () => {
                getDownloadURL(uploadTask.snapshot.ref).then(async (downloadURL) => {
                    await addDoc(collection(db, "files"), {
                        userId: user.uid,
                        name: file.name,
                        url: downloadURL,
                        createdAt: serverTimestamp(),
                    });
                    setFile(null);
                    setUploadProgress(0);
                });
            }
        );
    };

    const handleDeleteFile = async (fileId, fileUrl) => {
        if (window.confirm("¿Estás seguro de que quieres eliminar este archivo?")) {
            try {
                const fileRef = ref(storage, fileUrl);
                await deleteObject(fileRef);
                await deleteDoc(doc(db, "files", fileId));
            } catch (error) {
                console.error("Error al eliminar el archivo:", error);
                setError("Error al eliminar el archivo.");
            }
        }
    };

    const handleLogout = () => {
        signOut(auth).catch((error) => {
            console.error("Error al cerrar sesión:", error);
        });
    };

    if (currentView === 'chatbot') {
        return <ChatBot user={user} onBack={() => setCurrentView('dashboard')} />;
    }

    return (
        <div className="flex flex-col h-screen bg-gray-900 text-white">
            <header className="flex items-center justify-between p-4 bg-gray-800 border-b border-gray-700">
                <div className="flex items-center">
                    <DashboardIcon />
                    <h1 className="ml-2 text-xl font-bold">Panel de Archivos</h1>
                </div>
                <div className="flex items-center space-x-4">
                    <button
                        onClick={() => setCurrentView('chatbot')}
                        className="flex items-center px-4 py-2 font-medium text-white bg-indigo-600 rounded-md hover:bg-indigo-700 transition-colors"
                    >
                        <ChatIcon />
                        <span className="ml-2">Asistente Legal</span>
                    </button>
                    <button onClick={handleLogout} className="flex items-center px-4 py-2 font-medium text-white bg-red-600 rounded-md hover:bg-red-700 transition-colors">
                        <LogoutIcon />
                        <span className="ml-2">Cerrar Sesión</span>
                    </button>
                </div>
            </header>
            <main className="flex-1 p-8 overflow-y-auto">
                <section className="mb-8 p-6 bg-gray-800 rounded-lg shadow-lg">
                    <h2 className="text-2xl font-semibold mb-4">Subir Archivo</h2>
                    <div className="flex items-center space-x-4">
                        <label className="flex-1">
                            <input type="file" onChange={handleFileChange} className="hidden" />
                            <div className="flex items-center justify-center p-4 bg-gray-700 rounded-md cursor-pointer hover:bg-gray-600 transition-colors">
                                <UploadIcon />
                                <span className="font-medium text-gray-300">
                                    {file ? file.name : "Seleccionar archivo (Max. 5MB, solo PDF)"}
                                </span>
                            </div>
                        </label>
                        <button onClick={handleUpload} disabled={!file || uploadProgress > 0}
                            className="px-6 py-3 bg-indigo-600 rounded-md font-semibold hover:bg-indigo-700 disabled:bg-gray-600 transition-colors">
                            Subir
                        </button>
                    </div>
                    {error && <p className="mt-2 text-sm text-red-400">{error}</p>}
                    {uploadProgress > 0 && uploadProgress < 100 && (
                        <div className="w-full bg-gray-700 rounded-full mt-4">
                            <div className="bg-indigo-500 text-xs font-medium text-blue-100 text-center p-0.5 leading-none rounded-full"
                                style={{ width: `${uploadProgress}%` }}>
                                {`${uploadProgress.toFixed(0)}%`}
                            </div>
                        </div>
                    )}
                </section>

                <section className="p-6 bg-gray-800 rounded-lg shadow-lg">
                    <h2 className="text-2xl font-semibold mb-4">Mis Archivos</h2>
                    {loading ? (
                        <p className="text-gray-400">Cargando archivos...</p>
                    ) : filesList.length === 0 ? (
                        <p className="text-gray-400">Aún no has subido ningún archivo.</p>
                    ) : (
                        <ul className="space-y-4">
                            {filesList.map(fileItem => (
                                <li key={fileItem.id} className="flex items-center justify-between p-4 bg-gray-700 rounded-md">
                                    <div className="flex items-center">
                                        <FilePdfIcon />
                                        <a href={fileItem.url} target="_blank" rel="noopener noreferrer" className="ml-4 font-medium text-indigo-400 hover:underline">
                                            {fileItem.name}
                                        </a>
                                    </div>
                                    <button onClick={() => handleDeleteFile(fileItem.id, fileItem.url)} className="p-2 text-red-400 hover:text-red-500 rounded-full transition-colors">
                                        <DeleteIcon />
                                    </button>
                                </li>
                            ))}
                        </ul>
                    )}
                </section>
            </main>
        </div>
    );
}

// --- Password Reset Component ---
function PasswordResetComponent({ oobCode, onComplete }) {
    const [newPassword, setNewPassword] = useState('');
    const [confirmPassword, setConfirmPassword] = useState('');
    const [error, setError] = useState('');
    const [success, setSuccess] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [email, setEmail] = useState('');
    const [isValidCode, setIsValidCode] = useState(false);

    useEffect(() => {
        // Verificar que el código es válido y obtener el email
        const verifyCode = async () => {
            try {
                const email = await verifyPasswordResetCode(auth, oobCode);
                setEmail(email);
                setIsValidCode(true);
            } catch (err) {
                console.error("Error verificando código:", err);
                setError("El enlace de restablecimiento es inválido o ha expirado.");
            }
        };

        if (oobCode) {
            verifyCode();
        }
    }, [oobCode]);

    const handlePasswordReset = async (e) => {
        e.preventDefault();
        setError('');
        setSuccess('');

        // Validaciones
        if (!newPassword || !confirmPassword) {
            setError("Por favor, completa todos los campos.");
            return;
        }

        if (newPassword.length < 6) {
            setError("La contraseña debe tener al menos 6 caracteres.");
            return;
        }

        if (newPassword !== confirmPassword) {
            setError("Las contraseñas no coinciden.");
            return;
        }

        setIsLoading(true);

        try {
            await confirmPasswordReset(auth, oobCode, newPassword);
            setSuccess("¡Contraseña restablecida exitosamente! Ya puedes iniciar sesión con tu nueva contraseña.");
            
            // Esperar 3 segundos antes de redirigir al login
            setTimeout(() => {
                onComplete();
            }, 3000);
        } catch (err) {
            console.error("Error restableciendo contraseña:", err);
            
            switch (err.code) {
                case 'auth/weak-password':
                    setError("La contraseña es demasiado débil.");
                    break;
                case 'auth/expired-action-code':
                    setError("El enlace ha expirado. Solicita uno nuevo.");
                    break;
                case 'auth/invalid-action-code':
                    setError("El enlace es inválido o ya fue usado.");
                    break;
                default:
                    setError("Error al restablecer la contraseña. Inténtalo de nuevo.");
            }
        } finally {
            setIsLoading(false);
        }
    };

    if (!isValidCode && !error) {
        return (
            <div className="min-h-screen bg-gray-900 flex items-center justify-center p-4">
                <div className="bg-gray-800 p-8 rounded-lg shadow-2xl max-w-md w-full text-center">
                    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-400 mx-auto mb-4"></div>
                    <p className="text-white">Verificando enlace de restablecimiento...</p>
                </div>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-gray-900 flex items-center justify-center p-4">
            <div className="bg-gray-800 p-8 rounded-lg shadow-2xl max-w-md w-full">
                <h1 className="text-3xl font-bold text-white text-center mb-2">
                    Nueva Contraseña
                </h1>
                <p className="text-gray-400 text-center mb-6">
                    Restableciendo contraseña para: <span className="text-indigo-400">{email}</span>
                </p>

                {success ? (
                    <div className="text-center">
                        <div className="w-16 h-16 mx-auto mb-4 bg-green-600 rounded-full flex items-center justify-center">
                            <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                            </svg>
                        </div>
                        <p className="text-green-500 mb-4">{success}</p>
                        <p className="text-gray-400 text-sm">Redirigiendo al login...</p>
                    </div>
                ) : (
                    <form onSubmit={handlePasswordReset}>
                        <div className="mb-4">
                            <label className="block text-gray-400 text-sm font-bold mb-2" htmlFor="new-password">
                                Nueva Contraseña
                            </label>
                            <input
                                id="new-password"
                                type="password"
                                value={newPassword}
                                onChange={(e) => setNewPassword(e.target.value)}
                                className="shadow appearance-none border rounded w-full py-2 px-3 bg-gray-700 border-gray-600 text-white leading-tight focus:outline-none focus:ring-2 focus:ring-indigo-500"
                                placeholder="Mínimo 6 caracteres"
                                disabled={isLoading}
                            />
                        </div>

                        <div className="mb-6">
                            <label className="block text-gray-400 text-sm font-bold mb-2" htmlFor="confirm-password">
                                Confirmar Nueva Contraseña
                            </label>
                            <input
                                id="confirm-password"
                                type="password"
                                value={confirmPassword}
                                onChange={(e) => setConfirmPassword(e.target.value)}
                                className="shadow appearance-none border rounded w-full py-2 px-3 bg-gray-700 border-gray-600 text-white leading-tight focus:outline-none focus:ring-2 focus:ring-indigo-500"
                                placeholder="Confirma tu contraseña"
                                disabled={isLoading}
                            />
                        </div>

                        {error && (
                            <div className="mb-4 p-3 bg-red-900 border border-red-700 rounded">
                                <p className="text-red-300 text-sm">{error}</p>
                            </div>
                        )}

                        <div className="flex flex-col space-y-4">
                            <button 
                                type="submit"
                                disabled={isLoading}
                                className="w-full bg-indigo-600 hover:bg-indigo-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline transition-colors flex items-center justify-center"
                            >
                                {isLoading ? (
                                    <div className="flex items-center">
                                        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                                        Restableciendo...
                                    </div>
                                ) : (
                                    'Restablecer Contraseña'
                                )}
                            </button>

                            <button 
                                type="button"
                                onClick={onComplete}
                                disabled={isLoading}
                                className="w-full bg-gray-600 hover:bg-gray-700 disabled:bg-gray-500 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline transition-colors"
                            >
                                Volver al Login
                            </button>
                        </div>
                    </form>
                )}
            </div>
        </div>
    );
}

function AuthComponent() {
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [firstName, setFirstName] = useState('');
    const [lastName, setLastName] = useState('');
    const [isLogin, setIsLogin] = useState(true);
    const [error, setError] = useState('');
    const [successMessage, setSuccessMessage] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [showPassword, setShowPassword] = useState(false);
    const [passwordStrength, setPasswordStrength] = useState('');

    const validatePassword = (password) => {
        const minLength = password.length >= 6;
        const hasUpperCase = /[A-Z]/.test(password);
        const hasLowerCase = /[a-z]/.test(password);
        const hasNumber = /[0-9]/.test(password);
        const hasSpecialChar = /[!@#$%^&*(),.?":{}|<>]/.test(password);

        const strengthCriteria = [minLength, hasUpperCase, hasLowerCase, hasNumber, hasSpecialChar];
        const metCriteria = strengthCriteria.filter(criterion => criterion).length;

        if (metCriteria === 5) return 'Fuerte';
        if (metCriteria >= 3) return 'Media';
        return 'Débil';
    };

    const handlePasswordChange = (e) => {
        const newPassword = e.target.value;
        setPassword(newPassword);
        if (!isLogin) {
            setPasswordStrength(validatePassword(newPassword));
        }
    };

    const handleAuthAction = async (e) => {
        e.preventDefault();
        setError('');
        setSuccessMessage('');
        setIsLoading(true);

        if (!isLogin && (!firstName || !lastName)) {
            setError("Por favor, completa los campos de nombre y apellido.");
            setIsLoading(false);
            return;
        }

        if (!email || !password) {
            setError("Por favor, completa todos los campos requeridos.");
            setIsLoading(false);
            return;
        }

        if (!isLogin) {
            const passwordStrengthValue = validatePassword(password);
            if (passwordStrengthValue === 'Débil') {
                setError("La contraseña es demasiado débil. Debe tener al menos 6 caracteres, una mayúscula, una minúscula, un número y un carácter especial.");
                setIsLoading(false);
                return;
            }
        }

        try {
            if (isLogin) {
                await signInWithEmailAndPassword(auth, email, password);
                setSuccessMessage("¡Inicio de sesión exitoso! Redirigiendo...");
            } else {
                await createUserWithEmailAndPassword(auth, email, password);
                // Optionally store firstName and lastName in Firestore
                await setDoc(doc(db, "users", auth.currentUser.uid), {
                    firstName,
                    lastName,
                    email,
                    createdAt: serverTimestamp(),
                });
                setSuccessMessage("¡Cuenta creada con éxito! Ahora puedes iniciar sesión.");
                setIsLogin(true);
                setFirstName('');
                setLastName('');
                setPassword('');
                setPasswordStrength('');
            }
        } catch (err) {
            console.error("Error during authentication:", err);
            switch (err.code) {
                case 'auth/wrong-password':
                    setError("Contraseña incorrecta.");
                    break;
                case 'auth/user-not-found':
                    setError("No existe un usuario con este correo.");
                    break;
                case 'auth/email-already-in-use':
                    setError("Este correo ya está registrado.");
                    break;
                case 'auth/invalid-email':
                    setError("El correo electrónico no es válido.");
                    break;
                case 'auth/weak-password':
                    setError("La contraseña es demasiado débil.");
                    break;
                default:
                    setError("Error al procesar la solicitud. Inténtalo de nuevo.");
            }
        } finally {
            setIsLoading(false);
        }
    };

    const handleForgotPassword = async () => {
        if (!email) {
            setError("Por favor, introduce tu email para restablecer la contraseña.");
            return;
        }
        setIsLoading(true);
        try {
            await sendPasswordResetEmail(auth, email);
            setSuccessMessage("Se ha enviado un enlace para restablecer tu contraseña a tu correo.");
            setError('');
        } catch (err) {
            console.error("Error sending password reset email:", err);
            setError("Error al enviar el enlace de restablecimiento. Verifica el correo.");
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="min-h-screen bg-gray-900 flex items-center justify-center p-4">
            <div className="bg-gray-800 p-8 rounded-lg shadow-2xl max-w-md w-full">
                <h1 className="text-3xl font-bold text-white text-center mb-6">
                    {isLogin ? "Bienvenido" : "Crear Cuenta"}
                </h1>

                <form onSubmit={handleAuthAction}>
                    {!isLogin && (
                        <>
                            <div className="mb-4">
                                <label className="block text-gray-400 text-sm font-bold mb-2" htmlFor="firstName">
                                    Nombre
                                </label>
                                <input
                                    id="firstName"
                                    type="text"
                                    value={firstName}
                                    onChange={(e) => setFirstName(e.target.value)}
                                    className="shadow appearance-none border rounded w-full py-2 px-3 bg-gray-700 border-gray-600 text-white leading-tight focus:outline-none focus:ring-2 focus:ring-indigo-500"
                                    placeholder="Nombre"
                                    disabled={isLoading}
                                />
                            </div>
                            <div className="mb-4">
                                <label className="block text-gray-400 text-sm font-bold mb-2" htmlFor="lastName">
                                    Apellido
                                </label>
                                <input
                                    id="lastName"
                                    type="text"
                                    value={lastName}
                                    onChange={(e) => setLastName(e.target.value)}
                                    className="shadow appearance-none border rounded w-full py-2 px-3 bg-gray-700 border-gray-600 text-white leading-tight focus:outline-none focus:ring-2 focus:ring-indigo-500"
                                    placeholder="Apellido"
                                    disabled={isLoading}
                                />
                            </div>
                        </>
                    )}
                    <div className="mb-4">
                        <label className="block text-gray-400 text-sm font-bold mb-2" htmlFor="email">
                            Correo Electrónico
                        </label>
                        <input
                            id="email"
                            type="email"
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                            className="shadow appearance-none border rounded w-full py-2 px-3 bg-gray-700 border-gray-600 text-white leading-tight focus:outline-none focus:ring-2 focus:ring-indigo-500"
                            placeholder="Correo"
                            disabled={isLoading}
                        />
                    </div>
                    <div className="mb-6 relative">
                        <label className="block text-gray-400 text-sm font-bold mb-2" htmlFor="password">
                            Contraseña
                        </label>
                        <div className="flex items-center">
                            <input
                                id="password"
                                type={showPassword ? "text" : "password"}
                                value={password}
                                onChange={handlePasswordChange}
                                className="shadow appearance-none border rounded w-full py-2 px-3 bg-gray-700 border-gray-600 text-white leading-tight focus:outline-none focus:ring-2 focus:ring-indigo-500"
                                placeholder=""
                                disabled={isLoading}
                            />
                            <button
                                type="button"
                                onClick={() => setShowPassword(!showPassword)}
                                className="absolute right-3 text-gray-400 hover:text-white"
                                disabled={isLoading}
                            >
                                {showPassword ? <EyeOffIcon /> : <EyeIcon />}
                            </button>
                        </div>
                        {!isLogin && password && (
                            <div className="mt-2 text-sm">
                                <span className={`${
                                    passwordStrength === 'Fuerte' ? 'text-green-500' :
                                    passwordStrength === 'Media' ? 'text-yellow-500' :
                                    'text-red-500'
                                }`}>
                                    Fuerza de la contraseña: {passwordStrength}
                                </span>
                            </div>
                        )}
                    </div>

                    {error && <p className="text-red-500 text-xs italic mb-4">{error}</p>}
                    {successMessage && <p className="text-green-500 text-xs italic mb-4">{successMessage}</p>}

                    <div className="flex items-center justify-between">
                        <button
                            type="submit"
                            disabled={isLoading}
                            className="w-full bg-indigo-600 hover:bg-indigo-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline transition-colors"
                        >
                            {isLoading ? (
                                <div className="flex items-center justify-center">
                                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                                    {isLogin ? 'Iniciando...' : 'Registrando...'}
                                </div>
                            ) : (
                                isLogin ? 'Iniciar Sesión' : 'Registrarse'
                            )}
                        </button>
                    </div>
                </form>

                <div className="text-center mt-4">
                    {isLogin && (
                        <button
                            onClick={handleForgotPassword}
                            disabled={isLoading}
                            className="font-medium text-indigo-400 hover:text-indigo-500 disabled:text-gray-500 disabled:cursor-not-allowed"
                        >
                            ¿Olvidaste tu contraseña?
                        </button>
                    )}
                </div>

                <p className="mt-2 text-sm text-center text-gray-400">
                    {isLogin ? '¿No tienes una cuenta? ' : '¿Ya tienes una cuenta? '}
                    <button
                        onClick={() => {
                            setIsLogin(!isLogin);
                            setError('');
                            setSuccessMessage('');
                            setFirstName('');
                            setLastName('');
                            setPassword('');
                            setPasswordStrength('');
                            setShowPassword(false);
                        }}
                        disabled={isLoading}
                        className="font-medium text-indigo-400 hover:text-indigo-500 disabled:text-gray-500 disabled:cursor-not-allowed"
                    >
                        {isLogin ? 'Regístrate' : 'Inicia Sesión'}
                    </button>
                </p>
            </div>
        </div>
    );
}

// --- App Root Component ---
function App() {
    const [currentUser, setCurrentUser] = useState(null);
    const [loading, setLoading] = useState(true);
    const [resetMode, setResetMode] = useState(null);
    const [resetCode, setResetCode] = useState(null);

    useEffect(() => {
        // Verificar si estamos en modo de reset de contraseña
        const urlParams = new URLSearchParams(window.location.search);
        const mode = urlParams.get('mode');
        const oobCode = urlParams.get('oobCode');

        if (mode === 'resetPassword' && oobCode) {
            setResetMode('resetPassword');
            setResetCode(oobCode);
            setLoading(false);
            return;
        }

        // Listener de autenticación
        const unsubscribe = onAuthStateChanged(auth, (user) => {
            console.log("Auth state changed:", user ? "User logged in" : "No user"); // Debug log
            setCurrentUser(user);
            setLoading(false);
        });

        return () => unsubscribe();
    }, []);

    const handleResetComplete = () => {
        setResetMode(null);
        setResetCode(null);
        window.history.replaceState({}, document.title, window.location.pathname);
    };

    if (loading) {
        return (
            <div className="min-h-screen bg-gray-900 flex items-center justify-center">
                <div className="text-center">
                    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-400 mx-auto mb-4"></div>
                    <h1 className="text-4xl font-bold text-white">Cargando Saturn AI...</h1>
                </div>
            </div>
        );
    }

    if (resetMode === 'resetPassword') {
        return <PasswordResetComponent oobCode={resetCode} onComplete={handleResetComplete} />;
    }

    return currentUser ? <Dashboard user={currentUser} /> : <AuthComponent />;
}

export default App;