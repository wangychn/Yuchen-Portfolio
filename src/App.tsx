import { useState } from "react";
import type { ReactElement } from "react";
import Sidebar from "./components/Sidebar/Sidebar";
import About from "./pages/About/About";
import Experience from "./pages/Experience/Experience";
import Projects from "./pages/Projects/Projects";
import { SpeedInsights } from '@vercel/speed-insights/react';

type Page = "about" | "experience" | "projects";

function App() {
    const [page, setPage] = useState<Page>("about");

    // map of keys to pages for sidebar
    const pages: Record<Page, ReactElement> = {
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

            <SpeedInsights />
        </div>
    )

}

export default App;