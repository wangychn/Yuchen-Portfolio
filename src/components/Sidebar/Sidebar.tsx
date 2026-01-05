import styles from "./Sidebar.module.css";
import EmailIcon from "@mui/icons-material/Email";
import GitHubIcon from "@mui/icons-material/GitHub";
import LinkedInIcon from "@mui/icons-material/LinkedIn";
import CalendarMonthIcon from "@mui/icons-material/CalendarMonth";

import { personal_info } from "../../data/data";

type Props = {
    current: string;
    onChange: (page: string) => void;
}

const linkIconMap: Record<string, JSX.Element> = {
    GitHub: (
        <GitHubIcon
            sx={{ fontSize: 32, color: "#24292f" }}
        />
    ),
    LinkedIn: (
        <LinkedInIcon
            sx={{ fontSize: 32, color: "#24292f" }}
        />
    ),
    Email: (
        <EmailIcon
            sx={{ fontSize: 32, color: "#24292f" }}
        />
    ),
    Calendar: (
        <CalendarMonthIcon
            sx={{ fontSize: 32, color: "#24292f" }}
        />
    ),
};

const sidebarItems: { key: Page; label: string }[] = [
    { key: "about", label: "About" },
    { key: "experience", label: "Experience" },
    { key: "projects", label: "Projects" },
];

export default function Sidebar({ current, onChange }: Props) {
    return (
        <aside className={styles.sidebar}>
            <div className={styles.inner}>
                {/* Profile block */}
                <div className={styles.profile}>
                    <img
                        src={personal_info.profile_photo}
                        alt="Yuchen Wang"
                        className={styles.avatar}
                    />

                    <div className={styles.name}>Yuchen Wang</div>
                    <div className={styles.subtitle}>Student @ University of Michigan</div>

                    <div className={styles.iconRow} aria-label="Social links">
                        {personal_info.linksList.map((link) => {
                            const isEmail = link.label === "Email";
                            const href = isEmail ? `mailto:${link.href}` : link.href;

                            return (
                                <a
                                    key={link.label}
                                    className={styles.iconBtn}
                                    href={href}
                                    target={isEmail ? undefined : "_blank"}
                                    rel={isEmail ? undefined : "noreferrer"}
                                    title={link.label}
                                    aria-label={link.label}
                                >
                                    {linkIconMap[link.label]}
                                </a>
                            );
                        })}
                    </div>
                </div>

                <nav className={styles.nav} aria-label="Primary">
                    {sidebarItems.map((item) => {

                        return (
                            <button
                                key={item.key}
                                className={`${styles.navItem} ${current === item.key ? styles.active : ""}`}
                                onClick={() => onChange(item.key)}
                            >
                                <span>{item.label}</span>
                                <span className={styles.bridge} aria-hidden="true" />
                            </button>
                        );
                    })}
                </nav>

                <div className={styles.footer}>© 2019–2025 Yuchen Wang</div>
            </div>
        </aside >
    );
}