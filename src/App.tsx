import { useState } from "react";
import Sidebar from "./components/Sidebar/Sidebar";
import About from "./pages/About/About";
import Experience from "./pages/Experience/Experience";
import Projects from "./pages/Projects/Projects";


function App() {
    const [page, setPage] = useState("about");

    // map of keys to pages for sidebar
    const pages: Record<string, JSX.Element> = {
        about: <About />,
        experience: <Experience />,
        projects: <Projects />,
    };

    return (
        <div>
            <Sidebar current={page} onChange={setPage} />

            <div
                style={{
                    // height: "100vh",
                    marginLeft: "30%",
                    padding: "48px 56px",
                    background: "#ffffff",
                }}
            >
                {pages[page] ?? pages.about}
            </div>
        </div>
    )

}

export default App;