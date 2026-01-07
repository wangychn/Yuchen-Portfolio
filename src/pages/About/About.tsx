import {
    Box,
    Paper,
    TextField,
    Button,
    Typography,
    Divider,
    Stack,
    Link,
    Accordion,
    AccordionSummary,
    AccordionDetails,
    List,
    ListItem,
    ListItemText,
} from "@mui/material";

import ExpandMoreIcon from "@mui/icons-material/ExpandMore";

import { paragraphs, personal_info } from "../../data/data";
import CartPole from "../../components/Cartpole/Cartpole";

export default function About() {
    return (
        <Box>
            <Typography variant="h4" fontWeight={800} gutterBottom>
                About Me
            </Typography>

            {/* ABOUT ME HEADER + PICTURE + CARTPOLE */}
            <Stack direction="row" spacing={3} alignItems="flex-start">

                {/* IMAGE */}
                <Box
                    component="img"
                    src={personal_info.photo}
                    alt={personal_info.name}
                    sx={{
                        width: 190,
                        height: 280,
                        borderRadius: 3,
                        objectFit: "cover",
                    }}
                />

                {/* SHORT INTRO BLURB */}
                <Stack direction="column" spacing={3} alignItems="flex-start">
                    <Box>
                        <Typography sx={{
                            maxWidth: 640

                        }}>
                            {paragraphs.blurb}
                        </Typography>

                        <Stack direction="row" spacing={1} mt={1}>
                            {personal_info.linksList.map((l) => (
                                <Link
                                    key={l.href}
                                    href={l.href}
                                    target="_blank"
                                    underline="hover"
                                >
                                    {l.label}
                                </Link>
                            ))}
                        </Stack>
                    </Box>

                    {/* CARTPOLE */}
                    <CartPole />

                </Stack>
            </Stack>

            <Divider sx={{ my: 5 }} />

            <Typography
                variant="h5"
                fontWeight={700}
                color="primary"
                gutterBottom
            >
                Bio
            </Typography>

            <Typography sx={{ maxWidth: 760 }}>
                {paragraphs.bio}
            </Typography>

            <Divider sx={{ my: 5 }} />

            <Typography
                variant="h5"
                fontWeight={700}
                color="primary"
                gutterBottom
            >
                Currently working on
            </Typography>

            <Box sx={{ maxWidth: 760 }}>
                {(paragraphs.current_projects).map((item) => (
                    <Accordion key={item.title} elevation={0} disableGutters sx={{
                        border: "1px solid",
                        borderColor: "divider",
                        borderRadius: 2,
                        mb: 1,
                        "&:before": { display: "none" },
                    }}>
                        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                            <Typography fontWeight={650}>{item.title}</Typography>
                        </AccordionSummary>
                        <AccordionDetails>
                            {Array.isArray(item.details) ? (
                                <List dense sx={{ pt: 0 }}>
                                    {item.details.map((d: string) => (
                                        <ListItem key={d} sx={{ py: 0.25 }}>
                                            <ListItemText primary={d} />
                                        </ListItem>
                                    ))}
                                </List>
                            ) : (
                                <Typography>{item.details}</Typography>
                            )}
                        </AccordionDetails>
                    </Accordion>
                ))}
            </Box>

            <Divider sx={{ my: 5 }} />

            <Paper
                elevation={8}
                sx={{
                    mt: 6,
                    p: 3,
                    maxWidth: 760,
                    borderRadius: 3,
                    border: "1px solid",
                    borderColor: "divider",
                }}
            >

                <Typography color="primary" variant="h5" fontWeight={700} gutterBottom>
                    Contact me
                </Typography>

                <Typography color="text.secondary" sx={{ mb: 2 }}>
                    Feel free to reach out! I'm always happy to talk about anything :D
                </Typography>


                <Box
                    component="form"
                    onSubmit={(e) => {
                        e.preventDefault();
                        const form = new FormData(e.currentTarget);
                        const name = form.get("name");
                        const email = form.get("email");
                        const message = form.get("message");

                        window.location.href =
                            `mailto:wangyuch@umich.edu` +
                            `?subject=${encodeURIComponent(`Portfolio contact — ${name}`)}` +
                            `&body=${encodeURIComponent(
                                `Name: ${name}\nEmail: ${email}\n\n${message}`
                            )}`;
                    }}
                >
                    <Stack spacing={2}>
                        <Stack direction={{ xs: "column", sm: "row" }} spacing={2}>
                            <TextField name="name" label="Name" fullWidth required />
                            <TextField name="email" label="Email" type="email" fullWidth required />
                        </Stack>

                        <TextField
                            name="message"
                            label="Message"
                            multiline
                            minRows={4}
                            fullWidth
                            required
                        />

                        <Stack direction="row" justifyContent="flex-end">
                            <Button type="submit" variant="contained">
                                Send
                            </Button>
                        </Stack>
                    </Stack>
                </Box>
            </Paper>
        </Box >
    );
}