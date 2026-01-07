import { Box, Typography, Divider, Paper } from "@mui/material";

export default function Experience() {
    return (
        <Box>
            <Typography variant="h3" fontWeight={800} gutterBottom>
                Experiences
            </Typography>

            <Typography color="text.secondary" sx={{ maxWidth: 760, mb: 2 }}>
                A snapshot of my work and leadership across software, research, and applied projects.
                You can view my full resume below.
            </Typography>

            <Divider sx={{ my: 5 }} />

            {/* RESUME SECTION */}
            <Typography variant="h5" fontWeight={800} color="primary" gutterBottom>
                Resume
            </Typography>

            <Typography color="text.secondary" sx={{ mb: 2 }}>
                If the embedded viewer does not load, open it directly:{" "}
                <a href="/resume.pdf" target="_blank" rel="noreferrer">
                    resume.pdf
                </a>
            </Typography>

            <Paper
                elevation={0}
                sx={{
                    border: "1px solid",
                    borderColor: "divider",
                    borderRadius: 2,
                    overflow: "hidden",
                    backgroundColor: "background.paper",
                }}
            >
                <Box
                    component="iframe"
                    title="Resume PDF"
                    src="/resume.pdf"
                    sx={{
                        width: "82%",
                        height: { xs: 480, sm: 560, md: 760 },
                        border: 0,
                        display: "block",
                        margin: "0 auto",   // centers it
                        backgroundColor: "background.paper",
                    }}
                />
            </Paper>

            {/* Optional: placeholder for future timeline/entries */}
            {/* <Divider sx={{ my: 3 }} /> */}
            {/* <Typography variant="h6" fontWeight={750} gutterBottom>
                Experience Highlights
            </Typography>
            <Typography color="text.secondary" sx={{ maxWidth: 760 }}>
                (Coming soon) I’ll add concise, impact-focused summaries here with links to projects,
                papers, and demos.
            </Typography> */}
        </Box>
    );
}