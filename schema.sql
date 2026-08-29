--
-- PostgreSQL database dump
--

\restrict QlEMP2MtYg0cf4aC2xtgPLn0vq04lI7jgRguxR2KMLvVn2CPrS9yHwyArwzYy9A

-- Dumped from database version 17.11 (Homebrew)
-- Dumped by pg_dump version 17.11 (Homebrew)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: chunks; Type: TABLE; Schema: public; Owner: thanhhc
--

CREATE TABLE public.chunks (
    id bigint NOT NULL,
    show_id text NOT NULL,
    doc_title text NOT NULL,
    clause_id text,
    section_path text,
    page integer,
    kind text DEFAULT 'clause'::text NOT NULL,
    part text,
    content text NOT NULL,
    embed_text text NOT NULL,
    embedding public.vector(384),
    tsv tsvector GENERATED ALWAYS AS (to_tsvector('english'::regconfig, content)) STORED
);


ALTER TABLE public.chunks OWNER TO thanhhc;

--
-- Name: chunks_id_seq; Type: SEQUENCE; Schema: public; Owner: thanhhc
--

CREATE SEQUENCE public.chunks_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.chunks_id_seq OWNER TO thanhhc;

--
-- Name: chunks_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: thanhhc
--

ALTER SEQUENCE public.chunks_id_seq OWNED BY public.chunks.id;


--
-- Name: chunks id; Type: DEFAULT; Schema: public; Owner: thanhhc
--

ALTER TABLE ONLY public.chunks ALTER COLUMN id SET DEFAULT nextval('public.chunks_id_seq'::regclass);


--
-- Name: chunks chunks_pkey; Type: CONSTRAINT; Schema: public; Owner: thanhhc
--

ALTER TABLE ONLY public.chunks
    ADD CONSTRAINT chunks_pkey PRIMARY KEY (id);


--
-- Name: chunks_clause_id_idx; Type: INDEX; Schema: public; Owner: thanhhc
--

CREATE INDEX chunks_clause_id_idx ON public.chunks USING btree (clause_id);


--
-- Name: chunks_embedding_idx; Type: INDEX; Schema: public; Owner: thanhhc
--

CREATE INDEX chunks_embedding_idx ON public.chunks USING hnsw (embedding public.vector_cosine_ops);


--
-- Name: chunks_show_id_idx; Type: INDEX; Schema: public; Owner: thanhhc
--

CREATE INDEX chunks_show_id_idx ON public.chunks USING btree (show_id);


--
-- Name: chunks_tsv_idx; Type: INDEX; Schema: public; Owner: thanhhc
--

CREATE INDEX chunks_tsv_idx ON public.chunks USING gin (tsv);


--
-- PostgreSQL database dump complete
--

\unrestrict QlEMP2MtYg0cf4aC2xtgPLn0vq04lI7jgRguxR2KMLvVn2CPrS9yHwyArwzYy9A

