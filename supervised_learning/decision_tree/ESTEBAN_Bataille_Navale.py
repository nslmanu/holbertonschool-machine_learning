import random
import socket
import sys
import pygame

# Initialisation de Pygame
pygame.init()

# --- CONSTANTES DE DESIGN ---
NB_COLS = 12
NB_LIGNES = 10
TAILLE_CASE = 40
MARGE_HAUT = 60
MARGE_GAUCHE = 60

LARGEUR_FENETRE = (NB_COLS * TAILLE_CASE * 2) + MARGE_HAUT * 3
HAUTEUR_FENETRE = (NB_LIGNES * TAILLE_CASE) + MARGE_HAUT + 140

# Palette "Cyber-Marine" ultra soignée
COULEUR_FOND = (5, 15, 30)
COULEUR_PANNEAU = (15, 35, 65)
COULEUR_GRILLE = (30, 80, 140)
COULEUR_CASE = (230, 240, 250)
COULEUR_SURVOL = (0, 210, 255)
COULEUR_TEXTE = (245, 247, 250)
COULEUR_EAU = (40, 110, 170)
COULEUR_TOUCHE = (235, 60, 60)
COULEUR_COULE = (90, 100, 110)
COULEUR_BATEAU = (40, 180, 110)
COULEUR_SELECTION = (255, 200, 0)
COULEUR_BOUTON = (20, 90, 160)
COULEUR_BOUTON_HOVER = (0, 150, 255)
COULEUR_ACTIF = (0, 180, 255)
COULEUR_NEON_CYAN = (0, 220, 255)

fenetre = pygame.display.set_mode((LARGEUR_FENETRE, HAUTEUR_FENETRE))
pygame.display.set_caption("Bataille Navale - Édition Ultime")

police_titre = pygame.font.SysFont("Segoe UI", 36, bold=True)
police_texte = pygame.font.SysFont("Segoe UI", 18, bold=True)
police_petit = pygame.font.SysFont("Segoe UI", 14, bold=True)

CHIFFRES = [str(i) for i in range(1, 13)]
LETTRES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

LISTE_BATEAUX_MODELE = [
    {"nom": "Porte-avions", "taille": 5},
    {"nom": "Croiseur", "taille": 4},
    {"nom": "Contre-torpilleur", "taille": 3},
    {"nom": "Torpilleur", "taille": 3},
    {"nom": "Sous-marin", "taille": 2},
]


class GrilleNavale:

  def __init__(self):
    self.plateau = [[" " for _ in range(NB_COLS)] for _ in range(NB_LIGNES)]
    self.bateaux = []

  def est_placement_valide(self, cases):
    for l, c in cases:
      if not (0 <= l < NB_LIGNES and 0 <= c < NB_COLS):
        return False
      for dl in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
          nl, nc = l + dl, c + dc
          if 0 <= nl < NB_LIGNES and 0 <= nc < NB_COLS:
            if self.plateau[nl][nc] == "B":
              return False
    return True

  def placer_bateau_aleatoire(self, taille):
    place = False
    tentatives = 0
    while not place and tentatives < 2000:
      tentatives += 1
      horizontal = random.choice([True, False])
      if horizontal:
        c = random.randint(0, NB_COLS - taille)
        l = random.randint(0, NB_LIGNES - 1)
        cases = [(l, c + i) for i in range(taille)]
      else:
        c = random.randint(0, NB_COLS - 1)
        l = random.randint(0, NB_LIGNES - taille)
        cases = [(l + i, c) for i in range(taille)]

      if self.est_placement_valide(cases):
        for casel, casec in cases:
          self.plateau[casel][casec] = "B"
        self.bateaux.append({"cases": cases, "touches": 0})
        place = True


def verifier_coule(grille, l, c):
  for b in grille.bateaux:
    if (l, c) in b["cases"]:
      b["touches"] += 1
      if b["touches"] == len(b["cases"]):
        return "Coulé !"
      return "Touché !"
  return "À l'eau !"


def appartient_a_bateau_coule(grille, l, c):
  for b in grille.bateaux:
    if (l, c) in b["cases"] and b["touches"] == len(b["cases"]):
      return True
  return False


# --- BULLES D'AMBIANCE ---
bulles = [{
    "x": random.randint(0, LARGEUR_FENETRE),
    "y": random.randint(0, HAUTEUR_FENETRE),
    "rayon": random.randint(2, 7),
    "vitesse": random.uniform(0.4, 1.2),
} for _ in range(30)]


def mettre_a_jour_et_dessiner_bulles():
  for bulle in bulles:
    bulle["y"] -= bulle["vitesse"]
    if bulle["y"] < -10:
      bulle["y"] = HAUTEUR_FENETRE + 10
      bulle["x"] = random.randint(0, LARGEUR_FENETRE)
    surf_bulle = pygame.Surface(
        (bulle["rayon"] * 2, bulle["rayon"] * 2), pygame.SRCALPHA
    )
    pygame.draw.circle(
        surf_bulle,
        (0, 200, 255, 45),
        (bulle["rayon"], bulle["rayon"]),
        bulle["rayon"],
    )
    fenetre.blit(
        surf_bulle, (bulle["x"] - bulle["rayon"], bulle["y"] - bulle["rayon"])
    )


# --- VARIABLES DE JEU & RESEAU ---
etat_jeu = "MENU"
mode_jeu = "1J"

joueur1 = GrilleNavale()
joueur2 = GrilleNavale()
tour_joueur = True
message_statut = ""
dernier_resultat_tour = ""
gagnant = ""

bateaux_placement = []
index_bateau_selectionne = None
premiere_extremite = None
ia_en_attente = False
temps_ia = 0

socket_reseau = None
serveur_socket = None
est_serveur = False
ip_saisie = ""
port_reseau = 5555


def obtenir_ip_locale():
  try:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ip = s.getsockname()[0]
    s.close()
    return ip
  except:
    return "127.0.0.1"


def initialiser_partie(mode):
  global joueur1, joueur2, etat_jeu, mode_jeu, bateaux_placement, index_bateau_selectionne, premiere_extremite, message_statut, dernier_resultat_tour, tour_joueur, gagnant
  mode_jeu = mode
  joueur1 = GrilleNavale()
  joueur2 = GrilleNavale()

  if mode_jeu == "1J":
    tailles_ia = [b["taille"] for b in LISTE_BATEAUX_MODELE]
    random.shuffle(tailles_ia)
    for t in tailles_ia:
      joueur2.placer_bateau_aleatoire(t)

  bateaux_placement = [
      {"nom": b["nom"], "taille": b["taille"], "place": False}
      for b in LISTE_BATEAUX_MODELE
  ]
  index_bateau_selectionne = 0
  premiere_extremite = None
  tour_joueur = True
  dernier_resultat_tour = ""
  gagnant = ""
  etat_jeu = "PLACEMENT"
  message_statut = "Placez vos flottes avec précision, Commandant."


horloge = pygame.time.Clock()

while True:
  souris_x, souris_y = pygame.mouse.get_pos()
  temps_actuel = pygame.time.get_ticks()

  for event in pygame.event.get():
    if event.type == pygame.QUIT:
      if socket_reseau:
        try:
          socket_reseau.close()
        except:
          pass
      if serveur_socket:
        try:
          serveur_socket.close()
        except:
          pass
      pygame.quit()
      sys.exit()

    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
      if socket_reseau:
        try:
          socket_reseau.close()
        except:
          pass
        socket_reseau = None
      if serveur_socket:
        try:
          serveur_socket.close()
        except:
          pass
        serveur_socket = None
      etat_jeu = "MENU"

    if etat_jeu == "MENU":
      if event.type == pygame.MOUSEBUTTONDOWN:
        if (
            LARGEUR_FENETRE // 2 - 160
            <= souris_x
            <= LARGEUR_FENETRE // 2 + 160
            and 230 <= souris_y <= 295
        ):
          initialiser_partie("1J")
        elif (
            LARGEUR_FENETRE // 2 - 160
            <= souris_x
            <= LARGEUR_FENETRE // 2 + 160
            and 315 <= souris_y <= 380
        ):
          initialiser_partie("2J_LOCAL")
        elif (
            LARGEUR_FENETRE // 2 - 160
            <= souris_x
            <= LARGEUR_FENETRE // 2 + 160
            and 400 <= souris_y <= 465
        ):
          etat_jeu = "RESEAU_MENU"

    elif etat_jeu == "RESEAU_MENU":
      if event.type == pygame.MOUSEBUTTONDOWN:
        if (
            LARGEUR_FENETRE // 2 - 170
            <= souris_x
            <= LARGEUR_FENETRE // 2 - 10
            and 320 <= souris_y <= 380
        ):
          est_serveur = True
          try:
            serveur_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            serveur_socket.setsockopt(
                socket.SOL_SOCKET, socket.SO_REUSEADDR, 1
            )
            serveur_socket.bind(("", port_reseau))
            serveur_socket.listen(1)
            etat_jeu = "ATTENTE_RESEAU"
            message_statut = (
                "En attente d'un adversaire sur le réseau local..."
            )
          except:
            message_statut = "Erreur de création du serveur."
        elif (
            LARGEUR_FENETRE // 2 + 10
            <= souris_x
            <= LARGEUR_FENETRE // 2 + 170
            and 320 <= souris_y <= 380
        ):
          est_serveur = False
          ip_saisie = ""
          etat_jeu = "SAISIE_IP"
        elif (
            LARGEUR_FENETRE // 2 - 100
            <= souris_x
            <= LARGEUR_FENETRE // 2 + 100
            and 480 <= souris_y <= 535
        ):
          etat_jeu = "MENU"

    elif etat_jeu == "SAISIE_IP":
      if event.type == pygame.KEYDOWN:
        if event.key == pygame.K_RETURN:
          if ip_saisie.strip() != "":
            try:
              socket_reseau = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
              socket_reseau.connect((ip_saisie.strip(), port_reseau))
              mode_jeu = "2J_RESEAU"
              initialiser_partie("2J_RESEAU")
            except:
              message_statut = "Impossible de se connecter à cette IP !"
        elif event.key == pygame.K_BACKSPACE:
          ip_saisie = ip_saisie[:-1]
        else:
          if len(ip_saisie) < 20:
            ip_saisie += event.unicode

    elif etat_jeu == "PLACEMENT":
      if event.type == pygame.MOUSEBUTTONDOWN:
        decalage_droite = MARGE_GAUCHE * 2 + NB_COLS * TAILLE_CASE
        for i, b in enumerate(bateaux_placement):
          rect_b = pygame.Rect(decalage_droite, MARGE_HAUT + i * 50, 300, 42)
          if rect_b.collidepoint(souris_x, souris_y) and not b["place"]:
            index_bateau_selectionne = i
            premiere_extremite = None

        if (
            MARGE_GAUCHE
            <= souris_x
            <= MARGE_GAUCHE + NB_COLS * TAILLE_CASE
            and MARGE_HAUT
            <= souris_y
            <= MARGE_HAUT + NB_LIGNES * TAILLE_CASE
        ):
          if index_bateau_selectionne is not None:
            b_actif = bateaux_placement[index_bateau_selectionne]
            if not b_actif["place"]:
              c = (souris_x - MARGE_GAUCHE) // TAILLE_CASE
              l = (souris_y - MARGE_HAUT) // TAILLE_CASE
              taille_voulue = b_actif["taille"]

              if premiere_extremite is None:
                premiere_extremite = (l, c)
              else:
                l1, c1 = premiere_extremite
                l2, c2 = l, c
                cases = []
                if l1 == l2:
                  pas = 1 if c2 >= c1 else -1
                  cases = [(l1, col) for col in range(c1, c2 + pas, pas)]
                elif c1 == c2:
                  pas = 1 if l2 >= l1 else -1
                  cases = [(lig, c1) for lig in range(l1, l2 + pas, pas)]

                if len(cases) == taille_voulue and joueur1.est_placement_valide(
                    cases
                ):
                  for casel, casec in cases:
                    joueur1.plateau[casel][casec] = "B"
                  joueur1.bateaux.append({"cases": cases, "touches": 0})
                  b_actif["place"] = True
                  premiere_extremite = None
                  index_bateau_selectionne = None
                  for idx, b in enumerate(bateaux_placement):
                    if not b["place"]:
                      index_bateau_selectionne = idx
                      break

                  if all(b["place"] for b in bateaux_placement):
                    if mode_jeu == "1J":
                      etat_jeu = "JEU"
                      message_statut = "Feu à volonté !"
                    elif mode_jeu == "2J_LOCAL":
                      etat_jeu = "PLACEMENT_J2"
                      message_statut = "Tour du Joueur 2 : Placez vos navires."
                      bateaux_placement = [
                          {"nom": b["nom"], "taille": b["taille"], "place": False}
                          for b in LISTE_BATEAUX_MODELE
                      ]
                      index_bateau_selectionne = 0
                else:
                  premiere_extremite = None

      rect_alea = pygame.Rect(
          decalage_droite, HAUTEUR_FENETRE - 100, 300, 45
      )
      if rect_alea.collidepoint(souris_x, souris_y) and event.type == pygame.MOUSEBUTTONDOWN:
        joueur1 = GrilleNavale()
        tailles = [b["taille"] for b in LISTE_BATEAUX_MODELE]
        random.shuffle(tailles)
        for t in tailles:
          joueur1.placer_bateau_aleatoire(t)
        for b in bateaux_placement:
          b["place"] = True

        if mode_jeu == "1J":
          etat_jeu = "JEU"
        elif mode_jeu == "2J_LOCAL":
          etat_jeu = "PLACEMENT_J2"
          bateaux_placement = [
              {"nom": b["nom"], "taille": b["taille"], "place": False}
              for b in LISTE_BATEAUX_MODELE
          ]
          index_bateau_selectionne = 0

    elif etat_jeu == "PLACEMENT_J2":
      if event.type == pygame.MOUSEBUTTONDOWN:
        decalage_droite = MARGE_GAUCHE * 2 + NB_COLS * TAILLE_CASE
        for i, b in enumerate(bateaux_placement):
          rect_b = pygame.Rect(decalage_droite, MARGE_HAUT + i * 50, 300, 42)
          if rect_b.collidepoint(souris_x, souris_y) and not b["place"]:
            index_bateau_selectionne = i
            premiere_extremite = None

        if (
            MARGE_GAUCHE
            <= souris_x
            <= MARGE_GAUCHE + NB_COLS * TAILLE_CASE
            and MARGE_HAUT
            <= souris_y
            <= MARGE_HAUT + NB_LIGNES * TAILLE_CASE
        ):
          if index_bateau_selectionne is not None:
            b_actif = bateaux_placement[index_bateau_selectionne]
            if not b_actif["place"]:
              c = (souris_x - MARGE_GAUCHE) // TAILLE_CASE
              l = (souris_y - MARGE_HAUT) // TAILLE_CASE
              taille_voulue = b_actif["taille"]

              if premiere_extremite is None:
                premiere_extremite = (l, c)
              else:
                l1, c1 = premiere_extremite
                l2, c2 = l, c
                cases = []
                if l1 == l2:
                  pas = 1 if c2 >= c1 else -1
                  cases = [(l1, col) for col in range(c1, c2 + pas, pas)]
                elif c1 == c2:
                  pas = 1 if l2 >= l1 else -1
                  cases = [(lig, c1) for lig in range(l1, l2 + pas, pas)]

                if len(cases) == taille_voulue and joueur2.est_placement_valide(
                    cases
                ):
                  for casel, casec in cases:
                    joueur2.plateau[casel][casec] = "B"
                  joueur2.bateaux.append({"cases": cases, "touches": 0})
                  b_actif["place"] = True
                  premiere_extremite = None
                  index_bateau_selectionne = None
                  for idx, b in enumerate(bateaux_placement):
                    if not b["place"]:
                      index_bateau_selectionne = idx
                      break

                  if all(b["place"] for b in bateaux_placement):
                    etat_jeu = "TRANSITION_J1"
                    message_statut = "Prêt ? Passez l'ordinateur"
                else:
                  premiere_extremite = None

      rect_alea = pygame.Rect(
          decalage_droite, HAUTEUR_FENETRE - 100, 300, 45
      )
      if rect_alea.collidepoint(souris_x, souris_y) and event.type == pygame.MOUSEBUTTONDOWN:
        joueur2 = GrilleNavale()
        tailles = [b["taille"] for b in LISTE_BATEAUX_MODELE]
        random.shuffle(tailles)
        for t in tailles:
          joueur2.placer_bateau_aleatoire(t)
        for b in bateaux_placement:
          b["place"] = True
        etat_jeu = "TRANSITION_J1"
        message_statut = "Prêt ? Passez l'ordinateur"

    elif etat_jeu == "TRANSITION_J1":
      if event.type == pygame.MOUSEBUTTONDOWN:
        etat_jeu = "JEU"
        message_statut = "Au tour du Joueur 1 !"

    elif etat_jeu == "JEU":
      if mode_jeu in ["1J", "2J_LOCAL"]:
        if event.type == pygame.MOUSEBUTTONDOWN:
          decalage_ia = MARGE_GAUCHE * 2 + NB_COLS * TAILLE_CASE
          if (
              decalage_ia <= souris_x <= decalage_ia + NB_COLS * TAILLE_CASE
              and MARGE_HAUT <= souris_y <= MARGE_HAUT + NB_LIGNES * TAILLE_CASE
          ):
            c = (souris_x - decalage_ia) // TAILLE_CASE
            l = (souris_y - MARGE_HAUT) // TAILLE_CASE

            if joueur2.plateau[l][c] in [" ", "B"]:
              if joueur2.plateau[l][c] == "B":
                joueur2.plateau[l][c] = "X"
                res = verifier_coule(joueur2, l, c)
                if mode_jeu == "2J_LOCAL":
                  dernier_resultat_tour = (
                      f"Joueur 1 a tiré en {LETTRES[l]}{CHIFFRES[c]} : {res}"
                  )
                else:
                  message_statut = f"Tir en {LETTRES[l]}{CHIFFRES[c]} : {res}"
              else:
                joueur2.plateau[l][c] = "O"
                if mode_jeu == "2J_LOCAL":
                  dernier_resultat_tour = (
                      f"Joueur 1 a tiré en {LETTRES[l]}{CHIFFRES[c]} : À l'eau !"
                  )
                else:
                  message_statut = (
                      f"Tir en {LETTRES[l]}{CHIFFRES[c]} : À l'eau !"
                  )

              victoire = all(
                  b["touches"] == len(b["cases"]) for b in joueur2.bateaux
              )
              if victoire:
                gagnant = (
                    "Joueur 1 a triomphé !"
                    if mode_jeu == "2J_LOCAL"
                    else "Victoire éclatante !"
                )
                etat_jeu = "FIN"
              else:
                if mode_jeu == "1J":
                  tour_joueur = False
                  ia_en_attente = True
                  temps_ia = temps_actuel + 1200
                elif mode_jeu == "2J_LOCAL":
                  etat_jeu = "TRANSITION_TOUR"

    elif etat_jeu == "TRANSITION_TOUR":
      if event.type == pygame.MOUSEBUTTONDOWN:
        etat_jeu = "JEU_J2"
        dernier_resultat_tour = ""
        message_statut = "Au tour du Joueur 2 !"

    elif etat_jeu == "JEU_J2":
      if event.type == pygame.MOUSEBUTTONDOWN:
        if (
            MARGE_GAUCHE
            <= souris_x
            <= MARGE_GAUCHE + NB_COLS * TAILLE_CASE
            and MARGE_HAUT
            <= souris_y
            <= MARGE_HAUT + NB_LIGNES * TAILLE_CASE
        ):
          c = (souris_x - MARGE_GAUCHE) // TAILLE_CASE
          l = (souris_y - MARGE_HAUT) // TAILLE_CASE

          if joueur1.plateau[l][c] in [" ", "B"]:
            if joueur1.plateau[l][c] == "B":
              joueur1.plateau[l][c] = "X"
              res = verifier_coule(joueur1, l, c)
              dernier_resultat_tour = (
                  f"Joueur 2 a tiré en {LETTRES[l]}{CHIFFRES[c]} : {res}"
              )
            else:
              joueur1.plateau[l][c] = "O"
              dernier_resultat_tour = (
                  f"Joueur 2 a tiré en {LETTRES[l]}{CHIFFRES[c]} : À l'eau !"
              )

            victoire = all(
                b["touches"] == len(b["cases"]) for b in joueur1.bateaux
            )
            if victoire:
              gagnant = "Joueur 2 a triomphé !"
              etat_jeu = "FIN"
            else:
              etat_jeu = "TRANSITION_TOUR_J1"

    elif etat_jeu == "TRANSITION_TOUR_J1":
      if event.type == pygame.MOUSEBUTTONDOWN:
        etat_jeu = "JEU"
        dernier_resultat_tour = ""
        message_statut = "Au tour du Joueur 1 !"

    elif etat_jeu == "FIN":
      if event.type == pygame.MOUSEBUTTONDOWN:
        etat_jeu = "MENU"

  if etat_jeu == "ATTENTE_RESEAU" and serveur_socket:
    try:
      serveur_socket.settimeout(0.1)
      conn, addr = serveur_socket.accept()
      socket_reseau = conn
      serveur_socket.close()
      serveur_socket = None
      initialiser_partie("2J_RESEAU")
    except:
      pass

  if etat_jeu == "JEU" and mode_jeu == "1J" and not tour_joueur and ia_en_attente:
    if temps_actuel >= temps_ia:
      coups_possibles = [
          (l, c)
          for l in range(NB_LIGNES)
          for c in range(NB_COLS)
          if joueur1.plateau[l][c] in [" ", "B"]
      ]
      if coups_possibles:
        l_ia, c_ia = random.choice(coups_possibles)
        if joueur1.plateau[l_ia][c_ia] == "B":
          joueur1.plateau[l_ia][c_ia] = "X"
          res = verifier_coule(joueur1, l_ia, c_ia)
          message_statut = f"L'IA a tiré en {LETTRES[l_ia]}{CHIFFRES[c_ia]} : {res}"
        else:
          joueur1.plateau[l_ia][c_ia] = "O"
          message_statut = (
              f"L'IA a tiré en {LETTRES[l_ia]}{CHIFFRES[c_ia]} : À l'eau !"
          )

        victoire_ia = all(
            b["touches"] == len(b["cases"]) for b in joueur1.bateaux
        )
        if victoire_ia:
          gagnant = "L'IA l'emporte !"
          etat_jeu = "FIN"
        else:
          tour_joueur = True
          ia_en_attente = False

  # --- RENDU GRAPHIQUE ---
  fenetre.fill(COULEUR_FOND)
  mettre_a_jour_et_dessiner_bulles()

  if etat_jeu == "MENU":
    # Cadre décoratif du titre
    titre_rect = pygame.Rect(
        LARGEUR_FENETRE // 2 - 240, 100, 480, 80
    )
    pygame.draw.rect(fenetre, COULEUR_PANNEAU, titre_rect, border_radius=15)
    pygame.draw.rect(
        fenetre, COULEUR_NEON_CYAN, titre_rect, width=2, border_radius=15
    )

    txt_titre = police_titre.render("BATAILLE NAVALE", True, COULEUR_NEON_CYAN)
    fenetre.blit(
        txt_titre,
        (
            titre_rect.centerx - txt_titre.get_width() // 2,
            titre_rect.centery - txt_titre.get_height() // 2,
        ),
    )

    boutons_menu = [
        ("1 Joueur vs IA", 230),
        ("2 Joueurs Local", 315),
        ("2 Joueurs Réseau", 400),
    ]
    for label, y_pos in boutons_menu:
      rect_b = pygame.Rect(LARGEUR_FENETRE // 2 - 160, y_pos, 320, 65)
      survole = rect_b.collidepoint(souris_x, souris_y)
      couleur = COULEUR_BOUTON_HOVER if survole else COULEUR_BOUTON

      pygame.draw.rect(fenetre, couleur, rect_b, border_radius=12)
      pygame.draw.rect(
          fenetre,
          COULEUR_NEON_CYAN if survole else COULEUR_GRILLE,
          rect_b,
          width=2,
          border_radius=12,
      )

      txt = police_texte.render(label, True, COULEUR_TEXTE)
      fenetre.blit(
          txt,
          (
              rect_b.centerx - txt.get_width() // 2,
              rect_b.centery - txt.get_height() // 2,
          ),
      )

  elif etat_jeu == "RESEAU_MENU":
    txt_t = police_titre.render("MODE 2 JOUEURS (RÉSEAU)", True, COULEUR_NEON_CYAN)
    fenetre.blit(
        txt_t, (LARGEUR_FENETRE // 2 - txt_t.get_width() // 2, 70)
    )

    explications = [
        "Procédure de connexion à distance :",
        "1. L'hôte clique sur 'Héberger' (création du serveur).",
        "2. Le visiteur clique sur 'Rejoindre' et entre l'adresse IP.",
        f"Votre IP locale actuelle : {obtenir_ip_locale()}",
    ]
    for idx, ligne in enumerate(explications):
      t_exp = police_petit.render(
          ligne, True, (190, 220, 250) if idx < 3 else (255, 215, 0)
      )
      fenetre.blit(t_exp, (90, 150 + idx * 30))

    r_heb = pygame.Rect(LARGEUR_FENETRE // 2 - 170, 320, 160, 60)
    r_rej = pygame.Rect(LARGEUR_FENETRE // 2 + 10, 320, 160, 60)
    r_ret = pygame.Rect(LARGEUR_FENETRE // 2 - 100, 480, 200, 55)

    for r, texte in [(r_heb, "Héberger"), (r_rej, "Rejoindre"), (r_ret, "Retour")]:
      surv = r.collidepoint(souris_x, souris_y)
      c_col = COULEUR_BOUTON_HOVER if surv else COULEUR_BOUTON
      pygame.draw.rect(fenetre, c_col, r, border_radius=10)
      pygame.draw.rect(
          fenetre,
          COULEUR_NEON_CYAN if surv else COULEUR_GRILLE,
          r,
          width=2,
          border_radius=10,
      )
      t_btn = police_texte.render(texte, True, COULEUR_TEXTE)
      fenetre.blit(
          t_btn,
          (
              r.centerx - t_btn.get_width() // 2,
              r.centery - t_btn.get_height() // 2,
          ),
      )

  elif etat_jeu == "SAISIE_IP":
    txt_s = police_titre.render("CONNEXION DISTANTE", True, COULEUR_NEON_CYAN)
    fenetre.blit(
        txt_s, (LARGEUR_FENETRE // 2 - txt_s.get_width() // 2, 140)
    )

    rect_input = pygame.Rect(LARGEUR_FENETRE // 2 - 175, 240, 350, 55)
    pygame.draw.rect(fenetre, COULEUR_PANNEAU, rect_input, border_radius=10)
    pygame.draw.rect(
        fenetre, COULEUR_NEON_CYAN, rect_input, width=2, border_radius=10
    )

    txt_ip = police_texte.render(ip_saisie + "_", True, COULEUR_TEXTE)
    fenetre.blit(
        txt_ip,
        (
            rect_input.x + 15,
            rect_input.centery - txt_ip.get_height() // 2,
        ),
    )

    txt_info = police_petit.render(
        "Tapez l'adresse IP puis Entrée (Échap pour annuler)",
        True,
        (180, 210, 240),
    )
    fenetre.blit(
        txt_info, (LARGEUR_FENETRE // 2 - txt_info.get_width() // 2, 320)
    )

  elif etat_jeu == "ATTENTE_RESEAU":
    txt_att = police_titre.render("RADAR ACTIF...", True, COULEUR_NEON_CYAN)
    fenetre.blit(
        txt_att, (LARGEUR_FENETRE // 2 - txt_att.get_width() // 2, 200)
    )
    txt_ip_info = police_texte.render(
        f"Transmettez cette IP à votre allié : {obtenir_ip_locale()}",
        True,
        (255, 215, 0),
    )
    fenetre.blit(
        txt_ip_info,
        (LARGEUR_FENETRE // 2 - txt_ip_info.get_width() // 2, 260),
    )
    txt_esc = police_petit.render(
        "Appuyez sur ÉCHAP pour annuler", True, (150, 180, 210)
    )
    fenetre.blit(
        txt_esc, (LARGEUR_FENETRE // 2 - txt_esc.get_width() // 2, 330)
    )

  elif etat_jeu in [
      "TRANSITION_J1",
      "TRANSITION_TOUR",
      "TRANSITION_TOUR_J1",
  ]:
    # Fenêtre de transition futuriste
    trans_rect = pygame.Rect(
        LARGEUR_FENETRE // 2 - 280, 180, 560, 200
    )
    pygame.draw.rect(fenetre, COULEUR_PANNEAU, trans_rect, border_radius=15)
    pygame.draw.rect(
        fenetre, COULEUR_NEON_CYAN, trans_rect, width=2, border_radius=15
    )

    if dernier_resultat_tour:
      txt_res = police_texte.render(dernier_resultat_tour, True, (255, 120, 120))
      fenetre.blit(
          txt_res,
          (trans_rect.centerx - txt_res.get_width() // 2, trans_rect.y + 35),
      )

    prochain_joueur = (
        "Passez l'ordinateur au Joueur 1"
        if etat_jeu == "TRANSITION_TOUR_J1"
        else "Passez l'ordinateur"
    )
    txt_tr = police_titre.render(prochain_joueur, True, COULEUR_SELECTION)
    fenetre.blit(
        txt_tr,
        (trans_rect.centerx - txt_tr.get_width() // 2, trans_rect.y + 80),
    )

    txt_clic = police_petit.render(
        "[ Cliquez n'importe où pour valider ]", True, COULEUR_TEXTE
    )
    fenetre.blit(
        txt_clic,
        (trans_rect.centerx - txt_clic.get_width() // 2, trans_rect.y + 145),
    )

  else:
    txt_statut = police_texte.render(message_statut, True, COULEUR_NEON_CYAN)
    fenetre.blit(
        txt_statut,
        (LARGEUR_FENETRE // 2 - txt_statut.get_width() // 2, 15),
    )


  def dessiner_grille(ox, oy, grille_obj, est_grille_adversaire=False):
    rect_p = pygame.Rect(
        ox - 15,
        oy - 15,
        NB_COLS * TAILLE_CASE + 30,
        NB_LIGNES * TAILLE_CASE + 30,
    )
    pygame.draw.rect(fenetre, COULEUR_PANNEAU, rect_p, border_radius=12)
    pygame.draw.rect(fenetre, COULEUR_GRILLE, rect_p, width=2, border_radius=12)

    for c in range(NB_COLS):
      t = police_petit.render(CHIFFRES[c], True, COULEUR_NEON_CYAN)
      fenetre.blit(
          t,
          (
              ox + c * TAILLE_CASE + TAILLE_CASE // 2 - t.get_width() // 2,
              oy - 30,
          ),
      )
    for l in range(NB_LIGNES):
      t = police_petit.render(LETTRES[l], True, COULEUR_NEON_CYAN)
      fenetre.blit(
          t,
          (
              ox - 30,
              oy + l * TAILLE_CASE + TAILLE_CASE // 2 - t.get_height() // 2,
          ),
      )

    for l in range(NB_LIGNES):
      for c in range(NB_COLS):
        x = ox + c * TAILLE_CASE
        y = oy + l * TAILLE_CASE
        rect_case = pygame.Rect(x, y, TAILLE_CASE, TAILLE_CASE)

        couleur = COULEUR_CASE
        val = grille_obj.plateau[l][c]

        if val == "B" and not est_grille_adversaire:
          couleur = COULEUR_BATEAU
        elif val == "O":
          couleur = COULEUR_EAU
        elif val == "X":
          if appartient_a_bateau_coule(grille_obj, l, c):
            couleur = COULEUR_COULE
          else:
            couleur = COULEUR_TOUCHE
        elif (
            not est_grille_adversaire
            and premiere_extremite == (l, c)
            and etat_jeu in ["PLACEMENT", "PLACEMENT_J2"]
        ):
          couleur = COULEUR_SELECTION
        elif (
            rect_case.collidepoint(souris_x, souris_y)
            and etat_jeu in ["JEU", "JEU_J2"]
            and est_grille_adversaire
        ):
          couleur = COULEUR_SURVOL

        pygame.draw.rect(fenetre, couleur, rect_case)
        pygame.draw.rect(fenetre, COULEUR_GRILLE, rect_case, 1)


  if etat_jeu in ["PLACEMENT", "PLACEMENT_J2"]:
    dessiner_grille(
        MARGE_GAUCHE,
        MARGE_HAUT,
        joueur1 if etat_jeu == "PLACEMENT" else joueur2,
        False,
    )

    decalage_droite = MARGE_GAUCHE * 2 + NB_COLS * TAILLE_CASE
    rect_panneau_droite = pygame.Rect(
        decalage_droite - 15,
        MARGE_HAUT - 15,
        330,
        NB_LIGNES * TAILLE_CASE + 30,
    )
    pygame.draw.rect(fenetre, COULEUR_PANNEAU, rect_panneau_droite, border_radius=12)
    pygame.draw.rect(
        fenetre, COULEUR_GRILLE, rect_panneau_droite, width=2, border_radius=12
    )

    titre_p = (
        "Vos Bateaux :"
        if etat_jeu == "PLACEMENT"
        else "Flotte du Joueur 2 :"
    )
    txt_p_titre = police_texte.render(titre_p, True, COULEUR_NEON_CYAN)
    fenetre.blit(txt_p_titre, (decalage_droite, MARGE_HAUT))

    for i, b in enumerate(bateaux_placement):
      rect_b = pygame.Rect(decalage_droite, MARGE_HAUT + 35 + i * 50, 300, 42)
      if b["place"]:
        couleur_ligne = (25, 110, 75)
        statut_txt = "EN LIGNE"
      elif i == index_bateau_selectionne:
        couleur_ligne = COULEUR_ACTIF
        statut_txt = "EN COURS"
      else:
        couleur_ligne = (20, 50, 90)
        statut_txt = f"{b['taille']} cases"

      pygame.draw.rect(fenetre, couleur_ligne, rect_b, border_radius=8)
      pygame.draw.rect(
          fenetre,
          COULEUR_NEON_CYAN
          if i == index_bateau_selectionne
          else COULEUR_GRILLE,
          rect_b,
          width=1,
          border_radius=8,
      )

      texte_bateau = police_texte.render(
          f"{b['nom']} ({b['taille']})", True, COULEUR_TEXTE
      )
      texte_statut = police_petit.render(statut_txt, True, COULEUR_TEXTE)

      fenetre.blit(
          texte_bateau,
          (rect_b.x + 10, rect_b.centery - texte_bateau.get_height() // 2),
      )
      fenetre.blit(
          texte_statut,
          (
              rect_b.right - texte_statut.get_width() - 10,
              rect_b.centery - texte_statut.get_height() // 2,
          ),
      )

    rect_alea = pygame.Rect(decalage_droite, HAUTEUR_FENETRE - 100, 300, 45)
    surv_alea = rect_alea.collidepoint(souris_x, souris_y)
    couleur_alea = COULEUR_BOUTON_HOVER if surv_alea else COULEUR_BOUTON
    pygame.draw.rect(fenetre, couleur_alea, rect_alea, border_radius=8)
    pygame.draw.rect(
        fenetre,
        COULEUR_NEON_CYAN if surv_alea else COULEUR_GRILLE,
        rect_alea,
        width=2,
        border_radius=8,
    )
    txt_alea = police_texte.render("Placement Aléatoire", True, COULEUR_TEXTE)
    fenetre.blit(
        txt_alea,
        (
            rect_alea.centerx - txt_alea.get_width() // 2,
            rect_alea.centery - txt_alea.get_height() // 2,
        ),
    )

  elif etat_jeu == "JEU":
    dessiner_grille(MARGE_GAUCHE, MARGE_HAUT, joueur1, False)
    decalage_x = MARGE_GAUCHE * 2 + NB_COLS * TAILLE_CASE
    dessiner_grille(decalage_x, MARGE_HAUT, joueur2, True)

  elif etat_jeu == "JEU_J2":
    dessiner_grille(MARGE_GAUCHE, MARGE_HAUT, joueur1, True)
    decalage_x = MARGE_GAUCHE * 2 + NB_COLS * TAILLE_CASE
    dessiner_grille(decalage_x, MARGE_HAUT, joueur2, False)

  elif etat_jeu == "FIN":
    fin_rect = pygame.Rect(
        LARGEUR_FENETRE // 2 - 250, HAUTEUR_FENETRE - 90, 500, 60
    )
    pygame.draw.rect(fenetre, COULEUR_PANNEAU, fin_rect, border_radius=12)
    pygame.draw.rect(
        fenetre, COULEUR_SELECTION, fin_rect, width=2, border_radius=12
    )
    txt_fin = police_titre.render(gagnant, True, COULEUR_SELECTION)
    fenetre.blit(
        txt_fin,
        (
            fin_rect.centerx - txt_fin.get_width() // 2,
            fin_rect.centery - txt_fin.get_height() // 2,
        ),
    )

  pygame.display.flip()
  horloge.tick(60)